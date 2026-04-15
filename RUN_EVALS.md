If missing package:

````
yarn kbn bootstrap
````

Call Agent Builder converse API:

````
curl -X POST "http://localhost:5601/kyw/api/agent_builder/converse" -u "elastic:changeme" -H "kbn-xsrf: true" -H "Content-Type: application/json" -d '{"input": "Hello!"}'  
````


Example of command to run evals:

````
EVALUATION_CONNECTOR_ID=gemini25-pro-preview-connector \
DATASET_NAME="agent-builder: text-retrieval: wix-qa" \
EVALUATION_REPETITIONS=1 \
KBN_EVALS_SKIP_CONNECTOR_SETUP=true \
KBN_EVALS_EXECUTOR=phoenix \
node scripts/playwright test --config x-pack/platform/packages/shared/agent-builder/kbn-evals-suite-agent-builder/playwright.config.ts evals/external/external_dataset.spec.ts --project claude-sonnet-4-5-connector
````

points to: `.scout/servers/local.json`