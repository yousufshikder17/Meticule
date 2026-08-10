const SENSITIVE_KEY = /(?:authorization|cookie|password|passwd|secret|token|api[_-]?key|credential|private[_-]?key)/i;
const BEARER = /Bearer\s+[A-Za-z0-9._~+\/-]+=*/gi;

export function redact(value: unknown, depth = 0): unknown {
  if (depth > 8) return "[TRUNCATED]";
  if (typeof value === "string") return value.replace(BEARER, "Bearer [REDACTED]").slice(0, 4_000);
  if (value instanceof Error) return { name: value.name, message: redact(value.message, depth + 1) };
  if (Array.isArray(value)) return value.slice(0, 100).map((item) => redact(item, depth + 1));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value as Record<string, unknown>).slice(0, 100).map(([key, item]) => [key, SENSITIVE_KEY.test(key) ? "[REDACTED]" : redact(item, depth + 1)]));
  return value;
}
