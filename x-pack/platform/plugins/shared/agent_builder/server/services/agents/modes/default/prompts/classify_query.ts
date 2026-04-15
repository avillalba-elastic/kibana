/*
 * Copyright Elasticsearch B.V. and/or licensed to Elasticsearch B.V. under one
 * or more contributor license agreements. Licensed under the Elastic License
 * 2.0; you may not use this file except in compliance with the Elastic License
 * 2.0.
 */

import { cleanPrompt } from '@kbn/agent-builder-genai-utils/prompts';

export const getClassifyQueryPrompt = ({
  query,
  indexExplorerResult,
}: {
  query: string;
  indexExplorerResult: string | null;
}): string => {
  const sourcesSection = indexExplorerResult
    ? `## Available Elasticsearch sources\n${indexExplorerResult}`
    : `## Available Elasticsearch sources\nNot applicable — this query does not involve Elasticsearch data.`;

  return cleanPrompt(`You are a query complexity classifier. Your sole job is to classify the complexity of a user query so that an AI research agent can be given the right number of reasoning steps.

## Classification criteria

**Signal 1 — Query nature:**
- Raises complexity: correlating data across multiple dimensions, comparing entities, trends over time, anomaly detection, multi-entity comparisons, open-ended aggregations, queries with "across all", "between X and Y", "compare", "correlate"
- Lowers complexity: single-entity lookups, direct field retrievals, yes/no factual questions, simple counts

**Signal 2 — Data breadth (from the sources list below):**
- 1 source → lower complexity signal
- 2–3 sources → medium complexity signal
- 4+ sources or sources unavailable → higher complexity signal

## Classification table

| Sources | Query nature | → Tier |
|---------|-------------|--------|
| 1 | Simple lookup | simple |
| 1 | Aggregation / reasoning-heavy | medium |
| 2–3 | Any | medium |
| 4+ or N/A | Any | complex |

${sourcesSection}

## User query
${query}

Classify the query complexity. Output only the tier.`);
};
