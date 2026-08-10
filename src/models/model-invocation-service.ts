import type pg from "pg";
import type { Principal, AgentRecord } from "../db/types.js";
import type { ExecutionEngine } from "../execution/execution-engine.js";
import type { ModelProvider } from "./model-provider.js";
import type { CanonicalMessageSchema, CanonicalToolDefinitionSchema } from "./model-types.js";
import type { z } from "zod";
import { ModelProviderError, normalizeTransportError } from "./model-errors.js";
import { calculateCostMicrousd, type CostRates } from "./cost.js";
import { ConflictError } from "../domain/errors.js";

export class ModelBudgetExceededError extends Error {}
export class ModelStepLimitError extends Error {}
export interface ModelCandidate { provider: ModelProvider; target: CostRates & { model: string; maxOutputTokens: number; timeoutMs: number } }
export interface InvocationCommand { principal: Principal; runId: string; workerId: string; agent: AgentRecord; engine: ExecutionEngine; candidates: ModelCandidate[]; messages: z.infer<typeof CanonicalMessageSchema>[]; tools: z.infer<typeof CanonicalToolDefinitionSchema>[]; signal: AbortSignal; contextBuildId?: string }

interface Prepared { requestId: string; stepId: string }

export class ModelInvocationService {
  constructor(private readonly pool: pg.Pool) {}

  async invoke(command: InvocationCommand) {
    if (!command.candidates.length) throw new Error("At least one model candidate is required");
    const prepared = await this.prepare(command, crypto.randomUUID());
    const retry = command.agent.model.retryPolicy;
    let attemptNumber = 0;
    let lastError: ModelProviderError | null = null;
    for (let candidateIndex = 0; candidateIndex < command.candidates.length; candidateIndex += 1) {
      const candidate = command.candidates[candidateIndex]!;
      for (let localAttempt = 1; localAttempt <= retry.maxAttempts; localAttempt += 1) {
        attemptNumber += 1;
        let attemptId: string;
        try { await this.assertCanInvoke(command); attemptId = await this.createAttempt(command, prepared, candidate, attemptNumber, candidateIndex); }
        catch (error) { await this.interruptStep(prepared.stepId, error); throw error; }
        try {
          const result = await command.engine.executeTurn({ requestId: prepared.requestId, provider: candidate.provider, modelId: candidate.target.model, messages: command.messages, tools: command.tools, maxOutputTokens: candidate.target.maxOutputTokens, timeoutMs: candidate.target.timeoutMs, signal: command.signal });
          const cost = calculateCostMicrousd(result.response.usage, candidate.target);
          const completion = await this.complete(command, prepared, attemptId, candidate, result, cost);
          if (completion === "cancelled") throw new ConflictError("Cancellation detected after provider invocation");
          if (completion === "budget") throw new ModelBudgetExceededError("Model token or cost budget exceeded");
          return { ...result, costMicrousd: cost, stepId: prepared.stepId, attemptId, providerId: candidate.provider.id, attemptNumber };
        } catch (error) {
          if (error instanceof ModelBudgetExceededError || error instanceof ConflictError) throw error;
          const normalized = normalizeTransportError(error); lastError = normalized;
          await this.failAttempt(command, prepared.stepId, attemptId, candidate, normalized);
          if (!normalized.retryable) { await this.failStep(prepared.stepId, normalized); throw normalized; }
          const retrySame = localAttempt < retry.maxAttempts;
          const hasFallback = candidateIndex + 1 < command.candidates.length;
          if (!retrySame && !hasFallback) { await this.failStep(prepared.stepId, normalized); throw normalized; }
          try { await this.waitBackoff(command, localAttempt, retry.baseDelayMs, retry.maxDelayMs); }
          catch (waitError) { await this.interruptStep(prepared.stepId, waitError); throw waitError; }
        }
      }
    }
    const failure = lastError ?? new ModelProviderError("unavailable", "Configured model candidates were exhausted", false);
    await this.failStep(prepared.stepId, failure); throw failure;
  }

  private async prepare(command: InvocationCommand, requestId: string): Promise<Prepared> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [command.principal.tenantId, command.runId]);
      const run = result.rows[0]; this.validateRun(command, run);
      if (Number(run.current_step) >= command.agent.maximumSteps) throw new ModelStepLimitError("Maximum step limit reached");
      if (Number(run.input_tokens) + Number(run.output_tokens) + Number(run.reserved_child_tokens) >= Number(run.token_budget_limit)) throw new ModelBudgetExceededError("Token budget exhausted");
      if (Number(run.cost_microusd) + Number(run.reserved_child_cost_microusd) >= Number(run.cost_budget_limit_microusd)) throw new ModelBudgetExceededError("Cost budget exhausted");
      const sequence = Number(run.current_step) + 1;
      const first = command.candidates[0]!;
      const step = await client.query("INSERT INTO steps(tenant_id,run_id,sequence,kind,status,idempotency_key,attempt_count,input) VALUES($1,$2,$3,'model','running',$4,0,$5) RETURNING id", [command.principal.tenantId, command.runId, sequence, `model:${requestId}`, JSON.stringify({ provider: first.provider.id, model: first.target.model, engine: command.engine.id, fallbackCount: command.candidates.length - 1, contextBuildId: command.contextBuildId ?? null })]);
      await client.query("UPDATE runs SET current_step=$1,version=version+1,updated_at=now() WHERE id=$2", [sequence, command.runId]);
      await client.query("COMMIT"); return { requestId, stepId: step.rows[0].id as string };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async createAttempt(command: InvocationCommand, prepared: Prepared, candidate: ModelCandidate, attemptNumber: number, candidateIndex: number): Promise<string> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const runResult = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [command.principal.tenantId, command.runId]);
      this.validateRun(command, runResult.rows[0]);
      const attempt = await client.query("INSERT INTO model_attempts(tenant_id,run_id,step_id,request_id,provider_id,model_id,attempt_number,status,started_at,redacted_metadata) VALUES($1,$2,$3,$4,$5,$6,$7,'running',now(),$8) RETURNING id", [command.principal.tenantId, command.runId, prepared.stepId, prepared.requestId, candidate.provider.id, candidate.target.model, attemptNumber, JSON.stringify({ engine: command.engine.id, candidateIndex, contextBuildId: command.contextBuildId ?? null })]);
      await client.query("UPDATE steps SET attempt_count=$1 WHERE id=$2", [attemptNumber, prepared.stepId]);
      await client.query("COMMIT"); return attempt.rows[0].id as string;
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private validateRun(command: InvocationCommand, run: Record<string, unknown> | undefined): void {
    if (!run) throw new ConflictError("Run not found");
    if (run.cancellation_requested_at) throw new ConflictError("Cancellation requested");
    if (run.status !== "running" || run.lease_owner !== command.workerId || new Date(run.lease_expires_at as string | Date) <= new Date()) throw new ConflictError("Worker does not own a running lease");
  }

  private async assertCanInvoke(command: InvocationCommand): Promise<void> {
    if (command.signal.aborted) throw new ConflictError("Cancellation requested");
    const result = await this.pool.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2", [command.principal.tenantId, command.runId]);
    const run = result.rows[0]; this.validateRun(command, run);
    if (Number(run.input_tokens) + Number(run.output_tokens) + Number(run.reserved_child_tokens) >= Number(run.token_budget_limit) || Number(run.cost_microusd) + Number(run.reserved_child_cost_microusd) >= Number(run.cost_budget_limit_microusd)) throw new ModelBudgetExceededError("Model budget exhausted before retry");
  }

  private async complete(command: InvocationCommand, prepared: Prepared, attemptId: string, candidate: ModelCandidate, result: Awaited<ReturnType<ExecutionEngine["executeTurn"]>>, cost: number): Promise<"ok" | "cancelled" | "budget"> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const runResult = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [command.principal.tenantId, command.runId]);
      const run = runResult.rows[0]; if (!run) throw new ConflictError("Run disappeared");
      const usage = result.response.usage;
      const cancelled = Boolean(run.cancellation_requested_at) || run.status !== "running" || run.lease_owner !== command.workerId;
      const budget = Number(run.input_tokens) + Number(run.output_tokens) + Number(run.reserved_child_tokens) + usage.inputTokens + usage.outputTokens > Number(run.token_budget_limit) || Number(run.cost_microusd) + Number(run.reserved_child_cost_microusd) + cost > Number(run.cost_budget_limit_microusd);
      const status = cancelled ? "cancelled" : budget ? "failed" : "succeeded";
      await client.query("UPDATE model_attempts SET status=$1,completed_at=now(),updated_at=now(),stop_reason=$2,input_tokens=$3,output_tokens=$4,cached_tokens=$5,cost_microusd=$6,normalized_error_code=$7,redacted_metadata=$8,provider_request_id=$9 WHERE id=$10", [status, result.response.stopReason, usage.inputTokens, usage.outputTokens, usage.cachedTokens, cost, cancelled ? "cancelled" : budget ? "budget_exceeded" : null, JSON.stringify(result.response.metadata), result.response.providerRequestId ?? null, attemptId]);
      await client.query("INSERT INTO usage_records(tenant_id,run_id,step_id,provider,model,input_tokens,output_tokens,cost_microusd) VALUES($1,$2,$3,$4,$5,$6,$7,$8)", [command.principal.tenantId, command.runId, prepared.stepId, candidate.provider.id, candidate.target.model, usage.inputTokens, usage.outputTokens, cost]);
      await client.query("UPDATE runs SET input_tokens=input_tokens+$1,output_tokens=output_tokens+$2,cost_microusd=cost_microusd+$3,version=version+1,updated_at=now() WHERE id=$4", [usage.inputTokens, usage.outputTokens, cost, command.runId]);
      await client.query("UPDATE steps SET status=$1,output=$2,error_details=$3,finished_at=now() WHERE id=$4", [status, JSON.stringify(result.action), budget ? JSON.stringify({ code: "budget_exceeded" }) : null, prepared.stepId]);
      await client.query("COMMIT"); return cancelled ? "cancelled" : budget ? "budget" : "ok";
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async failAttempt(command: InvocationCommand, stepId: string, attemptId: string, candidate: ModelCandidate, error: ModelProviderError): Promise<void> {
    const usage = error.usage ?? { inputTokens: 0, outputTokens: 0, cachedTokens: 0 };
    const cost = calculateCostMicrousd(usage, candidate.target);
    const status = error.code === "cancelled" ? "cancelled" : error.code === "unknown_outcome" ? "unknown" : "failed";
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await client.query("UPDATE model_attempts SET status=$1,completed_at=now(),updated_at=now(),input_tokens=$2,output_tokens=$3,cached_tokens=$4,cost_microusd=$5,normalized_error_code=$6,redacted_metadata=$7 WHERE id=$8", [status, usage.inputTokens, usage.outputTokens, usage.cachedTokens, cost, error.code, JSON.stringify(error.redactedMetadata), attemptId]);
      if (usage.inputTokens || usage.outputTokens || cost) {
        await client.query("INSERT INTO usage_records(tenant_id,run_id,step_id,provider,model,input_tokens,output_tokens,cost_microusd) VALUES($1,$2,$3,$4,$5,$6,$7,$8)", [command.principal.tenantId, command.runId, stepId, candidate.provider.id, candidate.target.model, usage.inputTokens, usage.outputTokens, cost]);
        await client.query("UPDATE runs SET input_tokens=input_tokens+$1,output_tokens=output_tokens+$2,cost_microusd=cost_microusd+$3,version=version+1,updated_at=now() WHERE id=$4", [usage.inputTokens, usage.outputTokens, cost, command.runId]);
      }
      await client.query("COMMIT");
    } catch (failure) { await client.query("ROLLBACK").catch(() => undefined); throw failure; }
    finally { client.release(); }
  }

  private async failStep(stepId: string, error: ModelProviderError): Promise<void> {
    await this.pool.query("UPDATE steps SET status=$1,error_details=$2,finished_at=now() WHERE id=$3 AND status='running'", [error.code === "cancelled" ? "cancelled" : error.code === "unknown_outcome" ? "unknown" : "failed", JSON.stringify({ code: error.code, message: error.message, retryable: error.retryable }), stepId]);
  }

  private async interruptStep(stepId: string, error: unknown): Promise<void> {
    const cancelled = error instanceof ConflictError && /cancellation/i.test(error.message);
    await this.pool.query("UPDATE steps SET status=$1,error_details=$2,finished_at=now() WHERE id=$3 AND status='running'", [cancelled ? "cancelled" : "failed", JSON.stringify({ code: cancelled ? "cancelled" : "attempt_interrupted", message: error instanceof Error ? error.message : String(error) }), stepId]);
  }

  private async waitBackoff(command: InvocationCommand, attempt: number, baseMs: number, maxMs: number): Promise<void> {
    await this.assertCanInvoke(command);
    const delay = Math.min(maxMs, baseMs * (2 ** (attempt - 1))) + Math.floor(Math.random() * Math.max(1, Math.floor(baseMs / 4)));
    if (delay > 0) await new Promise((resolve) => setTimeout(resolve, delay));
    await this.assertCanInvoke(command);
  }
}
