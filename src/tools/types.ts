import type { z } from "zod";
import type pg from "pg";
import type { Principal } from "../db/types.js";

export type ToolRisk = "low" | "medium" | "high";
export type ToolRetrySafety = "pure" | "externally_idempotent" | "reconcilable" | "non_retryable";
export type ReconciliationResult =
  | { status: "succeeded"; output: unknown }
  | { status: "failed"; error: unknown }
  | { status: "pending" }
  | { status: "manual"; reason: string };
export interface ToolContext { pool: pg.Pool; principal: Principal; runId: string; idempotencyKey: string; signal: AbortSignal }
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: z.ZodType;
  outputSchema: z.ZodType;
  riskLevel: ToolRisk;
  authorization: { requiredRoles: string[] };
  approvalRequirement: "never" | "policy" | "always";
  timeoutMs: number;
  retryPolicy: { maxAttempts: number; retryableErrors: string[] };
  idempotency: "pure" | "keyed_side_effect";
  retrySafety: ToolRetrySafety;
  authorize?(input: unknown, context: ToolContext): Promise<void>;
  reconcile?(input: unknown, context: ToolContext): Promise<ReconciliationResult>;
  execute(input: unknown, context: ToolContext): Promise<unknown>;
}

export class ToolAuthorizationError extends Error {}
export class ToolTimeoutError extends Error {}
export class ToolExecutionConflictError extends Error {}
export class UnknownToolOutcomeError extends Error {}

export class ToolRegistry {
  private readonly tools = new Map<string, ToolDefinition>();
  register(tool: ToolDefinition): void {
    if (this.tools.has(tool.name)) throw new Error(`Duplicate tool: ${tool.name}`);
    this.tools.set(tool.name, tool);
  }
  get(name: string): ToolDefinition {
    const tool = this.tools.get(name); if (!tool) throw new Error(`Unsupported tool: ${name}`); return tool;
  }
  definitions(names: string[]): ToolDefinition[] { return names.map((name) => this.get(name)); }
  list(): Omit<ToolDefinition, "execute" | "authorize" | "reconcile" | "inputSchema" | "outputSchema">[] {
    return [...this.tools.values()].map(({ execute: _e, authorize: _a, reconcile: _r, inputSchema: _i, outputSchema: _o, ...metadata }) => metadata);
  }
}
