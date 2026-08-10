# Evaluation and regression

Stage 12 adds a persisted evaluation layer without adding an agent runtime. Every evaluation case creates an ordinary `queued` run. The normal PostgreSQL queue, lease worker, agent loop, tools, approvals, retries, recovery, budgets, cancellation, and terminal transitions remain authoritative. A separate worker-side scoring pass reads only terminal run evidence and cannot change a run's outcome.

## Versioned datasets

An evaluation suite contains ordered synthetic or reviewed cases, expectations, tags, and aggregate thresholds. Revisions create a new immutable suite version and archive the old version. Existing execution history continues to reference the exact version used.

Case expectations can evaluate:

- terminal status and exact or partial final output;
- selected tools, canonical arguments, forbidden tools, and approval use;
- retrieved document identities and citation source URIs;
- forbidden output markers and required recovery audit events;
- latency, token use, integer micro-USD cost, and step count.

The scorer records each dimension separately. An evaluation execution may complete successfully while its release gate is false; that distinction prevents a completed measurement job from being misreported as a passing candidate.

## Candidate identity

Runs freeze the complete validated agent configuration and agent version when created. Evaluation executions additionally record the candidate label and SHA-256 configuration hash. Prompt, provider, and model comparisons therefore remain attributable even if the agent definition is edited later. Comparisons require the same immutable suite version.

## Deterministic and model-dependent modes

`deterministic_ci` is accepted only by an explicitly constructed test-harness `EvaluationService`. The production API rejects it, and the deterministic model provider remains under `tests/`; it is not compiled into the runtime image. This mode supports stable CI regression gates through real PostgreSQL and the real worker lifecycle.

`model_dependent` queues runs against the agent's explicitly configured real provider. Results remain history rather than deterministic CI truth: model variability, provider availability, latency, and paid usage can change between executions. The platform never invokes paid providers automatically from the automated test suite.

## Roles and API

`evaluation_manager` creates/revises suites and starts executions. `evaluation_viewer` or `evaluation_manager` may inspect suites, execution history, measurements, and comparisons. Tenant predicates and PostgreSQL RLS apply to all five evaluation tables.

Implemented routes:

- `GET|POST /evaluation-suites`
- `GET /evaluation-suites/:id`
- `POST /evaluation-suites/:id/versions`
- `POST /evaluation-suites/:id/executions`
- `GET /evaluations`
- `GET /evaluations/:id`
- `GET /evaluations/:id/compare/:otherId`

Starting an execution only persists cases and queues canonical runs. It does not call a provider inline. Scoring is asynchronous and bounded to one completed case per worker polling cycle.

## Limits

The current scorer uses exact deterministic rules, not an LLM judge. Retrieval quality is required-document recall, not semantic relevance grading. Citation checks require both persisted source evidence and appearance of the expected URI in final output. Latency is durable run elapsed time and includes queue/worker overhead. Cost quality depends on provider-reported usage and configured rates. Dataset import/export, statistical confidence, baseline promotion, flaky-test classification, and hosted dashboards are not implemented.
