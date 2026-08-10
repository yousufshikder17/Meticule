export const MODEL_ERROR_CODES = ["authentication","authorization","invalid_request","model_not_found","rate_limited","unavailable","timeout","connection_failure","context_limit","content_filtered","malformed_response","cancelled","unknown_outcome"] as const;
export type ModelErrorCode = typeof MODEL_ERROR_CODES[number];

export class ModelProviderError extends Error {
  constructor(public readonly code: ModelErrorCode, message: string, public readonly retryable: boolean, public readonly redactedMetadata: Record<string, unknown> = {}, public readonly usage?: { inputTokens: number; outputTokens: number; cachedTokens: number }) {
    super(message); this.name = "ModelProviderError";
  }
}

export function mapHttpError(status: number, message = "Provider request failed"): ModelProviderError {
  if (status === 401) return new ModelProviderError("authentication", message, false);
  if (status === 403) return new ModelProviderError("authorization", message, false);
  if (status === 404) return new ModelProviderError("model_not_found", message, false);
  if (status === 429) return new ModelProviderError("rate_limited", message, true);
  if (status >= 500) return new ModelProviderError("unavailable", message, true);
  return new ModelProviderError("invalid_request", message, false);
}

export function normalizeTransportError(error: unknown): ModelProviderError {
  if (error instanceof ModelProviderError) return error;
  if (error instanceof DOMException && error.name === "AbortError") return new ModelProviderError("cancelled", "Provider call cancelled", false);
  if (error instanceof Error && /timeout/i.test(error.message)) return new ModelProviderError("timeout", "Provider call timed out", true);
  if (error instanceof TypeError) return new ModelProviderError("connection_failure", "Provider connection failed", true);
  return new ModelProviderError("unknown_outcome", "Provider outcome is unknown", false);
}
