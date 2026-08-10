import AjvModule from "ajv";
import { z } from "zod";
import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AgentRepository, RunRepository } from "../db/repositories.js";
import { validateComposition } from "./composition.js";
import { ExecutionEngineRegistry } from "./execution-engine-registry.js";
import { ProviderRegistry } from "../models/provider-registry.js";
import { ModelInvocationService, ModelBudgetExceededError, ModelStepLimitError } from "../models/model-invocation-service.js";
import { ToolExecutor } from "../tools/executor.js";
import { ToolRegistry, UnknownToolOutcomeError } from "../tools/types.js";
import { ConflictError } from "../domain/errors.js";
import { ApprovalService } from "../approvals/approval-service.js";
import { ContextBuilder } from "../context/context-builder.js";
import { PlanService } from "../planning/plan-service.js";
import type { RetrievalService } from "../retrieval/retrieval-service.js";
import type { ConnectorService } from "../connectors/connector-service.js";
import { OrchestrationService } from "../orchestration/orchestration-service.js";

const AjvConstructor = AjvModule as unknown as new (options?: object) => { compile(schema: object): ((data: unknown) => boolean) & { errors?: unknown }; errorsText(errors?: unknown): string };

export class AgentLoop {
  private readonly agents: AgentRepository; private readonly runs: RunRepository;
  private readonly invocation: ModelInvocationService; private readonly tools: ToolExecutor;
  private readonly approvals: ApprovalService; private readonly contexts: ContextBuilder;
  private readonly plans: PlanService; private readonly ajv = new AjvConstructor({ strict: false });
  private readonly orchestration: OrchestrationService;

  constructor(private readonly pool: pg.Pool, private readonly providers: ProviderRegistry, private readonly engines: ExecutionEngineRegistry, private readonly toolRegistry: ToolRegistry, retrieval: RetrievalService | null = null, connectors: ConnectorService | null = null) {
    this.agents = new AgentRepository(pool); this.runs = new RunRepository(pool);
    this.invocation = new ModelInvocationService(pool); this.tools = new ToolExecutor(pool, toolRegistry);
    this.approvals = new ApprovalService(pool, toolRegistry); this.contexts = new ContextBuilder(pool, retrieval, connectors); this.plans = new PlanService(pool); this.orchestration = new OrchestrationService(pool);
  }

  async execute(runId: string, workerId: string, roles: string[] = []): Promise<void> {
    const initial = await this.pool.query("SELECT tenant_id,created_by FROM runs WHERE id=$1", [runId]);
    if (!initial.rowCount) throw new Error("Run not found");
    const principal: Principal = { tenantId: initial.rows[0].tenant_id, userId: initial.rows[0].created_by, roles };
    try {
      while (true) {
        const run = await this.runs.get(principal.tenantId, runId);
        if (run.status === "cancelling") { await this.runs.workerTransition(runId, workerId, "cancelled"); return; }
        if (run.status !== "running") throw new ConflictError(`Run is not executable: ${run.status}`);
        const currentAgent = await this.agents.get(principal.tenantId, run.agentId);
        const agent: typeof currentAgent = { ...currentAgent, ...run.agentConfigurationSnapshot, version: run.agentVersion };
        const resumed = await this.tools.resumeApproved({ principal, runId, workerId });
        if (resumed.resumed) continue;
        await this.orchestration.assertReadyForTurn(principal.tenantId, runId);
        const composition = validateComposition(agent.composition);
        const engine = this.engines.get(composition.executionEngine);
        const provider = this.providers.get(agent.model.provider);
        const capabilities = provider.capabilities(agent.model.model);
        const definitions = this.toolRegistry.definitions(agent.allowedTools);
        const toolDefs = capabilities.toolCalling ? definitions.map((tool) => ({ name: tool.name, description: tool.description, inputSchema: z.toJSONSchema(tool.inputSchema) as Record<string, unknown> })) : [];
        const candidates = [{ provider, target: agent.model }, ...agent.model.fallbacks.map((target) => {
          const fallback = this.providers.get(target.provider); const fallbackCapabilities = fallback.capabilities(target.model);
          if (toolDefs.length && !fallbackCapabilities.toolCalling) throw new Error(`Fallback provider ${target.provider}/${target.model} lacks required tool-calling capability`);
          return { provider: fallback, target };
        })];
        const knownWindows = candidates.map((candidate) => candidate.provider.capabilities(candidate.target.model).contextWindow).filter((value): value is number => value !== null);
        const effectiveCapabilities = { ...capabilities, contextWindow: knownWindows.length ? Math.min(...knownWindows) : null };
        const context = await this.contexts.build({ tenantId: principal.tenantId, run, agent, capabilities: effectiveCapabilities, reservedOutputTokens: Math.max(...candidates.map((candidate) => candidate.target.maxOutputTokens)) });
        const invoked = await this.invocation.invoke({ principal, runId, workerId, agent, engine, candidates, messages: context.messages, tools: toolDefs, signal: new AbortController().signal, contextBuildId: context.id });
        const action = invoked.action;
        if (action.type === "call_tool") {
          if (!capabilities.toolCalling) throw new Error("Provider emitted an unsupported tool call");
          const current = await this.runs.get(principal.tenantId, runId);
          if (current.currentStep >= agent.maximumSteps) throw new ModelStepLimitError("Maximum step limit reached before tool execution");
          const command = { principal, runId, workerId, toolName: action.toolName, arguments: action.arguments, idempotencyKey: action.idempotencyKey ?? `${invoked.stepId}:${action.toolName}` };
          if (await this.approvals.proposeIfRequired(command)) return;
          await this.tools.execute(command); continue;
        }
        if (action.type === "final_answer") {
          const activePlan = await this.plans.active(principal.tenantId, runId);
          if (activePlan && activePlan.plan.tasks.some((task) => task.status !== "completed")) throw new ConflictError("Run cannot complete while required plan tasks remain incomplete");
          if (agent.outputSchema) { const validate = this.ajv.compile(agent.outputSchema); if (!validate(action.output)) throw new Error(`Final output schema validation failed: ${this.ajv.errorsText(validate.errors)}`); }
          await this.orchestration.assertCanComplete(principal.tenantId, runId);
          await this.runs.complete(runId, workerId, action.output); return;
        }
        if (action.type === "request_clarification") { await this.runs.workerTransition(runId, workerId, "paused", { code: "clarification_requested", question: action.question }); return; }
        if (action.type === "pause") {
          if (action.reason === "wait_for_children") {
            const result = await this.orchestration.pauseForChildren(principal, runId, workerId, action.reason);
            if (result === "none") throw new ConflictError("No delegated child runs exist to await");
            if (result === "ready") continue;
            return;
          }
          await this.runs.workerTransition(runId, workerId, "paused", { code: "model_pause", reason: action.reason }); return;
        }
        if (composition.planner !== "native") throw new ConflictError("Agent composition does not enable the native planner");
        await this.plans.revise({ principal, runId, workerId, stepId: invoked.stepId, plan: action.plan });
      }
    } catch (error) {
      const run = await this.runs.get(principal.tenantId, runId).catch(() => null);
      if (run?.leaseOwner === workerId && ["running", "cancelling"].includes(run.status)) {
        const target = run.cancellationRequestedAt ? "cancelled" : error instanceof UnknownToolOutcomeError ? "paused" : "failed";
        await this.runs.workerTransition(runId, workerId, target, { code: error instanceof ModelBudgetExceededError ? "budget_exceeded" : error instanceof ModelStepLimitError ? "step_limit" : error instanceof UnknownToolOutcomeError ? "reconciliation_required" : "execution_failed", message: error instanceof Error ? error.message : String(error) }).catch(() => undefined);
      }
      throw error;
    }
  }
}
