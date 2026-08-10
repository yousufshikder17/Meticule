# Model providers

Providers implement canonical messages, tool definitions/results, structured-output requests, usage, stop reasons, health checks, capabilities, and normalized errors. IDs are runtime-validated open strings. The executor resolves initialized instances through `ProviderRegistry`; `ProviderFactoryRegistry` builds them from validated configuration without a provider switch in lifecycle code.

| Adapter | Implemented | Automated evidence | Manual evidence |
|---|---:|---|---|
| deterministic test-only | tests only | complete loop/E2E | not a production provider |
| Ollama | yes | raw schema, registry, allowlist, E2E contract | `llama3.1:8b` completed locally |
| OpenAI-compatible | yes, overridable hooks | raw schema/mapping boundary | not manually verified |
| Anthropic | yes | raw content/tool schema | skipped: `ANTHROPIC_API_KEY` absent |
| Gemini | yes | raw content/function schema | skipped: `GEMINI_API_KEY` absent |

Capabilities are resolved per allowlisted model. Local models do not inherit tool or structured-output support merely because Ollama exposes those features. Missing cloud credentials prevent an enabled cloud adapter from starting; disabled cloud adapters require no key. Secrets are resolved from environment variables and are never stored in provider configuration, steps, attempts, or metadata.

Provider errors normalize to authentication, authorization, invalid request, model missing, rate limited, unavailable, timeout, connection, context limit, content filter, malformed response, cancelled, or unknown outcome, with explicit retryability. Stage 5 retries only retryable categories within configured bounds and uses only ordered explicit fallbacks. Each attempt is persisted separately. Capability-incompatible or unregistered fallbacks fail explicitly. Usage attached to failed provider errors is included in cumulative token/cost accounting.

To add a provider: implement one adapter plus mapping, Zod raw-response validation, error mapping, capability resolution, usage extraction, a factory, and contract tests. Do not edit run transitions, leasing, repositories, tool/approval/cancellation policy, or completion.

Wire formats were checked against the official Ollama chat/usage API, Anthropic Messages/tool-use API, Gemini function-calling API, and OpenAI Chat Completions conventions. Links are maintained here for adapter review: `https://docs.ollama.com/api/chat`, `https://docs.ollama.com/api/usage`, `https://platform.claude.com/docs/en/api/messages/create`, `https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview`, and `https://ai.google.dev/gemini-api/docs/function-calling`.

## Embedding providers

Embedding providers use a separate narrow contract because embeddings are retrieval components, not conversational execution engines. Stage 9 implements Ollama `/api/embed` with an explicit model allow-by-configuration, expected dimension, timeout/cancellation, health probe, raw-response validation, and no credentials for ordinary local use. Absence of `OLLAMA_EMBEDDING_MODEL` disables retrieval explicitly. Deterministic embeddings exist only under `tests/support` and are never a production fallback. The Stage 9 manual gate used `all-minilm` at 384 dimensions; other models require their actual dimension to be configured and verified.
