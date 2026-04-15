/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

import { END as _END_, START as _START_, StateGraph } from '@langchain/langgraph';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import type { BaseMessage } from '@langchain/core/messages';
import { HumanMessage } from '@langchain/core/messages';
import { z } from '@kbn/zod/v4';
import type { Logger } from '@kbn/core/server';
import type { InferenceChatModel } from '@kbn/inference-langchain';
import type { ResolvedAgentCapabilities } from '@kbn/agent-builder-common';
import { AgentExecutionErrorCode as ErrCodes } from '@kbn/agent-builder-common/agents';
import { createAgentExecutionError } from '@kbn/agent-builder-common/base/errors';
import type { AgentEventEmitter } from '@kbn/agent-builder-server';
import {
  createReasoningEvent,
  createToolCallMessage,
} from '@kbn/agent-builder-genai-utils/langchain';
import type { ToolManager } from '@kbn/agent-builder-server/runner';
import type { ResolvedConfiguration } from '../types';
import { convertError, isRecoverableError } from '../utils/errors';
import type { PromptFactory } from './prompts';
import { getRandomAnsweringMessage, getRandomThinkingMessage } from './i18n';
import { steps, tags } from './constants';
import type { StateType, ComplexityTier } from './state';
import { StateAnnotation } from './state';
import {
  processAnswerResponse,
  processResearchResponse,
  processToolNodeResponse,
} from './action_utils';
import { createAnswerAgentStructured } from './answer_agent_structured';
import {
  errorAction,
  handoverAction,
  isAgentErrorAction,
  isAnswerAction,
  isHandoverAction,
  isStructuredAnswerAction,
  isToolCallAction,
  isToolPromptAction,
} from './actions';
import type { ProcessedConversation } from '../utils/prepare_conversation';
import { getClassifyQueryPrompt } from './prompts/classify_query';

// number of successive recoverable errors we try to recover from before throwing
const MAX_ERROR_COUNT = 2;

const TIER_CYCLE_LIMITS: Record<ComplexityTier, number> = {
  simple: 3,
  medium: 6,
  complex: 10,
};

const tierSchema = z.object({
  tier: z.enum(['simple', 'medium', 'complex']).describe('The query complexity tier'),
});

export const createAgentGraph = ({
  chatModel,
  toolManager,
  configuration,
  capabilities,
  logger,
  events,
  structuredOutput = false,
  outputSchema,
  processedConversation,
  promptFactory,
}: {
  chatModel: InferenceChatModel;
  toolManager: ToolManager;
  capabilities: ResolvedAgentCapabilities;
  configuration: ResolvedConfiguration;
  logger: Logger;
  events: AgentEventEmitter;
  structuredOutput?: boolean;
  outputSchema?: Record<string, unknown>;
  processedConversation: ProcessedConversation;
  promptFactory: PromptFactory;
}) => {
  const init = async () => {
    return {};
  };

  const classifyQuery = async (state: StateType) => {
    const query = processedConversation.nextInput.message;

    // Signal 2: call index_explorer directly if available, to count accessible sources
    let indexExplorerResult: string | null = null;
    const indexExplorerTool = toolManager.list().find((t) => t.name === 'index_explorer');
    if (indexExplorerTool) {
      try {
        const result = await indexExplorerTool.invoke({ limit: 5 });
        indexExplorerResult = typeof result === 'string' ? result : JSON.stringify(result);
      } catch (err) {
        logger.debug(
          `[classifyQuery] index_explorer call failed, using text-only classification: ${err}`
        );
      }
    }

    try {
      const classifierWithOutput = chatModel
        .withStructuredOutput(tierSchema, { name: 'classify_query' })
        .withConfig({ tags: [tags.classifyAgent] });

      const { tier } = await classifierWithOutput.invoke([
        new HumanMessage(getClassifyQueryPrompt({ query, indexExplorerResult })),
      ]);

      // Cap at the current cycleLimit (the absolute max set by CYCLE_LIMIT in run_chat_agent)
      const cycleLimit = Math.min(TIER_CYCLE_LIMITS[tier], state.cycleLimit);
      logger.debug(`[classifyQuery] tier=${tier}, cycleLimit=${cycleLimit}`);

      return { complexityTier: tier, cycleLimit };
    } catch (err) {
      // On failure, keep the current cycleLimit unchanged (defaults to CYCLE_LIMIT = 15)
      logger.debug(
        `[classifyQuery] classification failed, keeping default cycleLimit=${state.cycleLimit}: ${err}`
      );
      return {};
    }
  };

  const researchAgent = async (state: StateType) => {
    const researcherModel = chatModel.bindTools(toolManager.list()).withConfig({
      tags: [tags.agent, tags.researchAgent],
    });

    if (state.mainActions.length === 0 && state.errorCount === 0) {
      events.emit(createReasoningEvent(getRandomThinkingMessage(), { transient: true }));
    }
    try {
      const response = await researcherModel.invoke(
        await promptFactory.getMainPrompt({
          actions: state.mainActions,
          cycleLimit: state.cycleLimit,
          currentCycle: state.currentCycle,
        })
      );

      const action = processResearchResponse(response);

      return {
        mainActions: [action],
        currentCycle: state.currentCycle + 1,
        errorCount: 0,
      };
    } catch (error) {
      const executionError = convertError(error);
      if (isRecoverableError(executionError)) {
        return {
          mainActions: [errorAction(executionError)],
          errorCount: state.errorCount + 1,
        };
      } else {
        throw executionError;
      }
    }
  };

  const researchAgentEdge = async (state: StateType) => {
    const lastAction = state.mainActions[state.mainActions.length - 1];

    if (isAgentErrorAction(lastAction)) {
      if (state.errorCount <= MAX_ERROR_COUNT) {
        return steps.researchAgent;
      } else {
        // max error count reached, stop execution by throwing
        throw lastAction.error;
      }
    } else if (isToolCallAction(lastAction)) {
      const maxCycleReached = state.currentCycle > state.cycleLimit;
      if (maxCycleReached) {
        return steps.prepareToAnswer;
      } else {
        return steps.executeTool;
      }
    } else if (isHandoverAction(lastAction)) {
      return steps.prepareToAnswer;
    }

    throw invalidState(`[researchAgentEdge] last action type was ${lastAction.type}}`);
  };

  const executeTool = async (state: StateType) => {
    const toolNode = new ToolNode<BaseMessage[]>(toolManager.list());

    const lastAction = state.mainActions[state.mainActions.length - 1];
    if (!isToolCallAction(lastAction)) {
      throw invalidState(
        `[executeTool] expected last action to be "tool_call" action, got "${lastAction.type}"`
      );
    }

    lastAction.tool_calls.forEach((toolCall) => toolManager.recordToolUse(toolCall.toolName));

    const toolCallMessage = createToolCallMessage(lastAction.tool_calls, lastAction.message);
    const toolNodeResult = await toolNode.invoke([toolCallMessage], {});
    const actions = processToolNodeResponse(toolNodeResult);

    return {
      mainActions: actions,
    };
  };

  const executeToolEdge = async (state: StateType) => {
    const lastAction = state.mainActions[state.mainActions.length - 1];
    if (isToolPromptAction(lastAction)) {
      return steps.handleToolInterrupt;
    }
    return steps.researchAgent;
  };

  const handleToolInterrupt = async (state: StateType) => {
    const lastAction = state.mainActions[state.mainActions.length - 1];
    if (!isToolPromptAction(lastAction)) {
      throw invalidState(`[handleToolInterrupt] last action type was ${lastAction.type}}`);
    }
    return {
      interrupted: true,
      prompts: lastAction.prompts.map((entry) => entry.prompt),
    };
  };

  const prepareToAnswer = async (state: StateType) => {
    const lastAction = state.mainActions[state.mainActions.length - 1];
    const maxCycleReached = state.currentCycle > state.cycleLimit;

    if (maxCycleReached && !isHandoverAction(lastAction)) {
      return {
        mainActions: [handoverAction('', true)],
      };
    } else {
      return {};
    }
  };

  const answeringModel = chatModel.withConfig({
    tags: [tags.agent, tags.answerAgent],
  });

  const answerAgent = async (state: StateType) => {
    if (state.answerActions.length === 0 && state.errorCount === 0) {
      events.emit(createReasoningEvent(getRandomAnsweringMessage(), { transient: true }));
    }
    try {
      const response = await answeringModel.invoke(
        await promptFactory.getAnswerPrompt({
          actions: state.mainActions,
          answerActions: state.answerActions,
        })
      );

      const action = processAnswerResponse(response);

      return {
        answerActions: [action],
        errorCount: 0,
      };
    } catch (error) {
      const executionError = convertError(error);
      if (isRecoverableError(executionError)) {
        return {
          answerActions: [errorAction(executionError)],
          errorCount: state.errorCount + 1,
        };
      } else {
        throw executionError;
      }
    }
  };

  const answerAgentStructured = createAnswerAgentStructured({
    chatModel,
    promptFactory,
    events,
    outputSchema,
    logger,
  });

  const answerAgentEdge = async (state: StateType) => {
    const lastAction = state.answerActions[state.answerActions.length - 1];

    if (isAgentErrorAction(lastAction)) {
      if (state.errorCount <= MAX_ERROR_COUNT) {
        return steps.answerAgent;
      } else {
        // max error count reached, stop execution by throwing
        throw lastAction.error;
      }
    } else if (isAnswerAction(lastAction) || isStructuredAnswerAction(lastAction)) {
      return steps.finalize;
    }

    // @ts-expect-error - lastAction.type is never because we cover all use cases.
    throw invalidState(`[answerAgentEdge] last action type was ${lastAction.type}}`);
  };

  const finalize = async (state: StateType) => {
    const answerAction = state.answerActions[state.answerActions.length - 1];
    if (isStructuredAnswerAction(answerAction)) {
      return {
        finalAnswer: answerAction.data,
      };
    } else if (isAnswerAction(answerAction)) {
      return {
        finalAnswer: answerAction.message,
      };
    } else {
      throw invalidState(`[finalize] expect answer action, got ${answerAction.type} instead.`);
    }
  };

  const selectedAnswerAgent = structuredOutput ? answerAgentStructured : answerAgent;

  // note: the node names are used in the event convertion logic, they should *not* be changed
  const graph = new StateGraph(StateAnnotation)
    // nodes
    .addNode(steps.init, init)
    .addNode(steps.classifyQuery, classifyQuery)
    .addNode(steps.researchAgent, researchAgent)
    .addNode(steps.executeTool, executeTool)
    .addNode(steps.handleToolInterrupt, handleToolInterrupt)
    .addNode(steps.prepareToAnswer, prepareToAnswer)
    .addNode(steps.answerAgent, selectedAnswerAgent)
    .addNode(steps.finalize, finalize)
    // edges
    .addEdge(_START_, steps.init)
    .addEdge(steps.init, steps.classifyQuery)
    .addEdge(steps.classifyQuery, steps.researchAgent)
    .addConditionalEdges(steps.researchAgent, researchAgentEdge, {
      [steps.researchAgent]: steps.researchAgent,
      [steps.executeTool]: steps.executeTool,
      [steps.prepareToAnswer]: steps.prepareToAnswer,
    })
    .addConditionalEdges(steps.executeTool, executeToolEdge, {
      [steps.researchAgent]: steps.researchAgent,
      [steps.handleToolInterrupt]: steps.handleToolInterrupt,
    })
    .addEdge(steps.handleToolInterrupt, _END_)
    .addEdge(steps.prepareToAnswer, steps.answerAgent)
    .addConditionalEdges(steps.answerAgent, answerAgentEdge, {
      [steps.answerAgent]: steps.answerAgent,
      [steps.finalize]: steps.finalize,
    })
    .addEdge(steps.finalize, _END_)
    .compile();

  return graph;
};

const invalidState = (message: string) => {
  return createAgentExecutionError(message, ErrCodes.invalidState, {});
};
