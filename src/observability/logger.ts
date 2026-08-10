import { redact } from "./redaction.js";

export type LogLevel = "info" | "warn" | "error";
export interface StructuredLogger { log(level: LogLevel, event: string, fields?: Record<string, unknown>): void }

export class JsonLogger implements StructuredLogger {
  constructor(private readonly sink: Pick<Console, "log" | "warn" | "error"> = console) {}
  log(level: LogLevel, event: string, fields: Record<string, unknown> = {}): void {
    const record = JSON.stringify({ timestamp: new Date().toISOString(), level, event, ...redact(fields) as Record<string, unknown> });
    const method = level === "info" ? "log" : level;
    this.sink[method](record);
  }
}

export class NullLogger implements StructuredLogger { log(): void {} }
export const logger = new JsonLogger();
