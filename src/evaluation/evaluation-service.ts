import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { agentConfigurationFromDatabaseRow } from "../db/repositories.js";
import { CreateEvaluationExecutionSchema, CreateEvaluationSuiteSchema, EvaluationExecutionListQuerySchema, EvaluationExpectationsSchema, EvaluationThresholdsSchema, type CreateEvaluationExecution, type CreateEvaluationSuite, type EvaluationThresholds } from "./evaluation-schema.js";
import { scoreEvaluationCase, type EvaluationTrace, type Measurement } from "./evaluation-scorer.js";

type Row = Record<string, unknown>;
const TERMINAL_RUNS = ["completed", "failed", "cancelled"];
const READ_ROLES = ["evaluation_viewer", "evaluation_manager"];

function requireRole(principal: Principal, roles: string[]): void {
  if (!roles.some((role) => principal.roles.includes(role))) throw new AuthorizationError(`Required role: ${roles.join(" or ")}`);
}

async function transaction<T>(pool: pg.Pool, work: (client: pg.PoolClient) => Promise<T>): Promise<T> {
  const client = await pool.connect();
  try { await client.query("BEGIN"); const result = await work(client); await client.query("COMMIT"); return result; }
  catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
  finally { client.release(); }
}

export class EvaluationService {
  constructor(private readonly pool: pg.Pool, private readonly options: { allowDeterministicCi?: boolean } = {}) {}

  async createSuite(principal: Principal, raw: CreateEvaluationSuite): Promise<Row> {
    requireRole(principal, ["evaluation_manager"]); const input = CreateEvaluationSuiteSchema.parse(raw);
    return transaction(this.pool, async (client) => {
      const suite = await client.query(
        `INSERT INTO evaluation_suites(tenant_id,version,created_by,name,description,thresholds)
         VALUES($1,1,$2,$3,$4,$5) RETURNING *`,
        [principal.tenantId, principal.userId, input.name, input.description, JSON.stringify(input.thresholds)],
      );
      await this.insertCases(client, principal.tenantId, suite.rows[0].id, input.cases);
      return suite.rows[0] as Row;
    });
  }

  async reviseSuite(principal: Principal, suiteId: string, raw: CreateEvaluationSuite): Promise<Row> {
    requireRole(principal, ["evaluation_manager"]); const input = CreateEvaluationSuiteSchema.parse(raw);
    return transaction(this.pool, async (client) => {
      const previous = await client.query("SELECT * FROM evaluation_suites WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, suiteId]);
      if (!previous.rowCount) throw new NotFoundError("Evaluation suite not found");
      if (previous.rows[0].status !== "active") throw new ConflictError("Only the active evaluation suite version can be revised");
      await client.query("UPDATE evaluation_suites SET status='archived' WHERE tenant_id=$1 AND id=$2", [principal.tenantId, suiteId]);
      const suite = await client.query(
        `INSERT INTO evaluation_suites(tenant_id,logical_id,version,supersedes_id,created_by,name,description,thresholds)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8) RETURNING *`,
        [principal.tenantId, previous.rows[0].logical_id, Number(previous.rows[0].version) + 1, suiteId, principal.userId, input.name, input.description, JSON.stringify(input.thresholds)],
      );
      await this.insertCases(client, principal.tenantId, suite.rows[0].id, input.cases);
      return suite.rows[0] as Row;
    });
  }

  async listSuites(principal: Principal): Promise<Row[]> {
    requireRole(principal, READ_ROLES);
    return (await this.pool.query("SELECT * FROM evaluation_suites WHERE tenant_id=$1 ORDER BY logical_id,version DESC", [principal.tenantId])).rows as Row[];
  }

  async getSuite(principal: Principal, suiteId: string): Promise<{ suite: Row; cases: Row[] }> {
    requireRole(principal, READ_ROLES);
    const suite = await this.pool.query("SELECT * FROM evaluation_suites WHERE tenant_id=$1 AND id=$2", [principal.tenantId, suiteId]);
    if (!suite.rowCount) throw new NotFoundError("Evaluation suite not found");
    const cases = await this.pool.query("SELECT * FROM evaluation_cases WHERE tenant_id=$1 AND suite_id=$2 ORDER BY ordinal", [principal.tenantId, suiteId]);
    return { suite: suite.rows[0] as Row, cases: cases.rows as Row[] };
  }

  async createExecution(principal: Principal, suiteId: string, raw: CreateEvaluationExecution): Promise<Row> {
    requireRole(principal, ["evaluation_manager"]); const input = CreateEvaluationExecutionSchema.parse(raw);
    if (input.mode === "deterministic_ci" && !this.options.allowDeterministicCi) throw new ConflictError("Deterministic CI providers are restricted to the test harness");
    return transaction(this.pool, async (client) => {
      const suite = await client.query("SELECT * FROM evaluation_suites WHERE tenant_id=$1 AND id=$2 AND status='active' FOR SHARE", [principal.tenantId, suiteId]);
      if (!suite.rowCount) throw new NotFoundError("Active evaluation suite not found");
      const cases = await client.query("SELECT * FROM evaluation_cases WHERE tenant_id=$1 AND suite_id=$2 ORDER BY ordinal", [principal.tenantId, suiteId]);
      if (!cases.rowCount) throw new ConflictError("Evaluation suite has no cases");
      const agent = await client.query("SELECT * FROM agents WHERE tenant_id=$1 AND id=$2 FOR SHARE", [principal.tenantId, input.agentId]);
      if (!agent.rowCount) throw new NotFoundError("Evaluation agent not found");
      const snapshot = agentConfigurationFromDatabaseRow(agent.rows[0] as Row); const snapshotHash = canonicalJsonHash(snapshot);
      const execution = await client.query(
        `INSERT INTO evaluation_executions(tenant_id,suite_id,agent_id,agent_version,agent_snapshot,agent_snapshot_hash,mode,candidate_label,status,case_count,created_by,started_at)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,'running',$9,$10,now()) RETURNING *`,
        [principal.tenantId, suiteId, input.agentId, agent.rows[0].version, JSON.stringify(snapshot), snapshotHash, input.mode, input.candidateLabel, cases.rowCount, principal.userId],
      );
      for (const evaluationCase of cases.rows) {
        const runId = crypto.randomUUID();
        await client.query(
          `INSERT INTO runs(id,tenant_id,agent_id,created_by,goal,root_run_id,token_budget_limit,cost_budget_limit_microusd,agent_version,agent_configuration_snapshot)
           VALUES($1,$2,$3,$4,$5,$1,$6,$7,$8,$9)`,
          [runId, principal.tenantId, input.agentId, principal.userId, evaluationCase.goal, snapshot.tokenBudget, snapshot.costBudgetMicrousd, agent.rows[0].version, JSON.stringify(snapshot)],
        );
        const caseRun = await client.query(
          `INSERT INTO evaluation_case_runs(tenant_id,execution_id,case_id,run_id,status,started_at)
           VALUES($1,$2,$3,$4,'pending',now()) RETURNING id`, [principal.tenantId, execution.rows[0].id, evaluationCase.id, runId],
        );
        await client.query(
          `INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES
           ($1,$2,'user',$3,'run.created',$4),($1,$2,'system','evaluation-service','evaluation.case_queued',$5)`,
          [principal.tenantId, runId, principal.userId, JSON.stringify({ status: "queued" }), JSON.stringify({ evaluationId: execution.rows[0].id, caseId: evaluationCase.id, caseRunId: caseRun.rows[0].id, mode: input.mode })],
        );
      }
      return execution.rows[0] as Row;
    });
  }

  async advanceNext(workerId: string): Promise<boolean> {
    return transaction(this.pool, async (client) => {
      const selected = await client.query(
        `SELECT ecr.*,ec.expectations,r.status AS run_status FROM evaluation_case_runs ecr
         JOIN evaluation_cases ec ON ec.tenant_id=ecr.tenant_id AND ec.id=ecr.case_id
         JOIN runs r ON r.tenant_id=ecr.tenant_id AND r.id=ecr.run_id
         WHERE ecr.status='pending' AND r.status IN ('completed','failed','cancelled')
         ORDER BY ecr.created_at,ecr.id FOR UPDATE OF ecr SKIP LOCKED LIMIT 1`,
      );
      if (!selected.rowCount) return false;
      const row = selected.rows[0];
      // Serialize scoring within one execution. Without this parent lock, two
      // scorers could each observe the other's uncommitted case as pending and
      // both skip the final aggregate transition.
      await client.query("SELECT id FROM evaluation_executions WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [row.tenant_id, row.execution_id]);
      await client.query("UPDATE evaluation_case_runs SET status='scoring' WHERE id=$1", [row.id]);
      const trace = await this.loadTrace(client, row.tenant_id, row.run_id);
      const score = scoreEvaluationCase(EvaluationExpectationsSchema.parse(row.expectations), trace);
      for (const metric of score.measurements) await this.insertMeasurement(client, row, metric);
      await client.query(
        "UPDATE evaluation_case_runs SET status=$1,metrics=$2,failures=$3,completed_at=now() WHERE id=$4",
        [score.passed ? "passed" : "failed", JSON.stringify(score.metrics), JSON.stringify(score.failures), row.id],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,'worker',$3,'evaluation.case_scored',$4)", [row.tenant_id, row.run_id, workerId, JSON.stringify({ evaluationId: row.execution_id, caseRunId: row.id, passed: score.passed })]);
      await this.finalizeIfReady(client, row.tenant_id, row.execution_id);
      return true;
    });
  }

  async getExecution(principal: Principal, executionId: string): Promise<{ execution: Row; caseRuns: Row[]; measurements: Row[] }> {
    requireRole(principal, READ_ROLES);
    const execution = await this.pool.query("SELECT * FROM evaluation_executions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, executionId]);
    if (!execution.rowCount) throw new NotFoundError("Evaluation execution not found");
    const caseRuns = await this.pool.query(`SELECT ecr.*,ec.name AS case_name,ec.ordinal,r.status AS run_status FROM evaluation_case_runs ecr JOIN evaluation_cases ec ON ec.tenant_id=ecr.tenant_id AND ec.id=ecr.case_id JOIN runs r ON r.tenant_id=ecr.tenant_id AND r.id=ecr.run_id WHERE ecr.tenant_id=$1 AND ecr.execution_id=$2 ORDER BY ec.ordinal`, [principal.tenantId, executionId]);
    const measurements = await this.pool.query("SELECT * FROM evaluation_measurements WHERE tenant_id=$1 AND execution_id=$2 ORDER BY metric_name,case_run_id", [principal.tenantId, executionId]);
    return { execution: execution.rows[0] as Row, caseRuns: caseRuns.rows as Row[], measurements: measurements.rows as Row[] };
  }

  async listExecutions(principal: Principal, rawQuery: unknown): Promise<Row[]> {
    requireRole(principal, READ_ROLES);
    const query = EvaluationExecutionListQuerySchema.parse(rawQuery);
    return (await this.pool.query(
      `SELECT * FROM evaluation_executions
       WHERE tenant_id=$1 AND ($2::uuid IS NULL OR suite_id=$2) AND ($3::uuid IS NULL OR agent_id=$3)
       AND ($4::text IS NULL OR mode=$4) ORDER BY created_at DESC,id LIMIT $5`,
      [principal.tenantId, query.suiteId ?? null, query.agentId ?? null, query.mode ?? null, query.limit],
    )).rows as Row[];
  }

  async compare(principal: Principal, leftId: string, rightId: string): Promise<Record<string, unknown>> {
    requireRole(principal, READ_ROLES);
    const result = await this.pool.query("SELECT * FROM evaluation_executions WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND status='completed'", [principal.tenantId, [leftId, rightId]]);
    if (result.rowCount !== 2) throw new ConflictError("Both evaluation executions must be completed in the same tenant");
    const left = result.rows.find((row) => row.id === leftId); const right = result.rows.find((row) => row.id === rightId);
    if (left.suite_id !== right.suite_id) throw new ConflictError("Evaluation comparisons require the same immutable suite version");
    const averages = async (id: string) => Object.fromEntries((await this.pool.query("SELECT metric_name,avg(numeric_value)::float8 AS value FROM evaluation_measurements WHERE tenant_id=$1 AND execution_id=$2 GROUP BY metric_name", [principal.tenantId, id])).rows.map((row) => [row.metric_name, Number(row.value)]));
    const leftMetrics = await averages(leftId); const rightMetrics = await averages(rightId);
    const names = [...new Set([...Object.keys(leftMetrics), ...Object.keys(rightMetrics)])].sort();
    return {
      left: this.candidateSummary(left), right: this.candidateSummary(right),
      metricDeltas: Object.fromEntries(names.map((name) => [name, (rightMetrics[name] ?? 0) - (leftMetrics[name] ?? 0)])),
    };
  }

  private async insertCases(client: pg.PoolClient, tenantId: string, suiteId: string, cases: CreateEvaluationSuite["cases"]): Promise<void> {
    for (const [index, evaluationCase] of cases.entries()) await client.query(
      `INSERT INTO evaluation_cases(tenant_id,suite_id,ordinal,name,goal,expectations,tags) VALUES($1,$2,$3,$4,$5,$6,$7)`,
      [tenantId, suiteId, index + 1, evaluationCase.name, evaluationCase.goal, JSON.stringify(evaluationCase.expectations), JSON.stringify(evaluationCase.tags)],
    );
  }

  private async loadTrace(client: pg.PoolClient, tenantId: string, runId: string): Promise<EvaluationTrace> {
    // A PostgreSQL client owns one protocol stream. Keep trace reads sequential so
    // scoring remains compatible with pg clients and tenant-scoped savepoints.
    const run = await client.query("SELECT * FROM runs WHERE tenant_id=$1 AND id=$2", [tenantId, runId]);
    const tools = await client.query("SELECT tool_name,validated_arguments,output FROM tool_executions WHERE tenant_id=$1 AND run_id=$2 ORDER BY created_at,id", [tenantId, runId]);
    const approvals = await client.query("SELECT tool_name FROM approvals WHERE tenant_id=$1 AND run_id=$2", [tenantId, runId]);
    const contextBuilds = await client.query("SELECT selected_document_chunk_ids FROM context_builds WHERE tenant_id=$1 AND run_id=$2", [tenantId, runId]);
    const events = await client.query("SELECT event_type FROM audit_events WHERE tenant_id=$1 AND run_id=$2", [tenantId, runId]);
    if (!run.rowCount || !TERMINAL_RUNS.includes(run.rows[0].status)) throw new ConflictError("Evaluation run is not terminal");
    const selectedChunkIds = [...new Set(contextBuilds.rows.flatMap((row) => row.selected_document_chunk_ids as string[]))];
    const documents = selectedChunkIds.length ? await client.query(
      `SELECT DISTINCT d.logical_id,d.source_uri FROM document_chunks dc JOIN documents d ON d.tenant_id=dc.tenant_id AND d.id=dc.document_id
       WHERE dc.tenant_id=$1 AND dc.id=ANY($2::uuid[])`, [tenantId, selectedChunkIds],
    ) : { rows: [] as Row[] };
    const toolCitationUris: string[] = [];
    for (const tool of tools.rows) {
      const results = tool.tool_name === "knowledge_search" && tool.output && typeof tool.output === "object" ? (tool.output as { results?: Array<{ citation?: { sourceUri?: string | null } }> }).results ?? [] : [];
      for (const item of results) if (item.citation?.sourceUri) toolCitationUris.push(item.citation.sourceUri);
    }
    const row = run.rows[0];
    return {
      run: { status: row.status, finalOutput: row.final_output, inputTokens: Number(row.input_tokens), outputTokens: Number(row.output_tokens), costMicrousd: Number(row.cost_microusd), currentStep: Number(row.current_step), createdAt: row.created_at, updatedAt: row.updated_at },
      toolExecutions: tools.rows.map((tool) => ({ toolName: tool.tool_name, validatedArguments: tool.validated_arguments })),
      approvalTools: approvals.rows.map((approval) => approval.tool_name),
      retrievedDocumentIds: documents.rows.map((document) => document.logical_id as string),
      citationSourceUris: [...new Set([...documents.rows.map((document) => document.source_uri as string | null).filter((value): value is string => Boolean(value)), ...toolCitationUris])],
      auditEvents: events.rows.map((event) => event.event_type),
    };
  }

  private async insertMeasurement(client: pg.PoolClient, row: Row, metric: Measurement): Promise<void> {
    await client.query(
      `INSERT INTO evaluation_measurements(tenant_id,execution_id,case_run_id,metric_name,numeric_value,unit,passed,details)
       VALUES($1,$2,$3,$4,$5,$6,$7,$8)`,
      [row.tenant_id, row.execution_id, row.id, metric.name, metric.value, metric.unit, metric.passed, JSON.stringify(metric.details)],
    );
  }

  private async finalizeIfReady(client: pg.PoolClient, tenantId: string, executionId: string): Promise<void> {
    const pending = await client.query("SELECT count(*)::int AS count FROM evaluation_case_runs WHERE tenant_id=$1 AND execution_id=$2 AND status IN ('pending','scoring')", [tenantId, executionId]);
    if (Number(pending.rows[0].count)) return;
    const execution = await client.query(`SELECT ee.*,es.thresholds FROM evaluation_executions ee JOIN evaluation_suites es ON es.tenant_id=ee.tenant_id AND es.id=ee.suite_id WHERE ee.tenant_id=$1 AND ee.id=$2 FOR UPDATE OF ee`, [tenantId, executionId]);
    const thresholds = EvaluationThresholdsSchema.parse(execution.rows[0].thresholds);
    const cases = await client.query("SELECT status,count(*)::int AS count FROM evaluation_case_runs WHERE tenant_id=$1 AND execution_id=$2 GROUP BY status", [tenantId, executionId]);
    const counts = Object.fromEntries(cases.rows.map((row) => [row.status, Number(row.count)]));
    const metrics = Object.fromEntries((await client.query("SELECT metric_name,avg(numeric_value)::float8 AS value FROM evaluation_measurements WHERE tenant_id=$1 AND execution_id=$2 GROUP BY metric_name", [tenantId, executionId])).rows.map((row) => [row.metric_name, Number(row.value)]));
    const passRate = (counts.passed ?? 0) / Number(execution.rows[0].case_count); const gateFailures = this.thresholdFailures(thresholds, passRate, metrics);
    const summary = { passRate, averages: metrics, gateFailures, thresholds, candidate: this.candidateSummary(execution.rows[0]) };
    await client.query(
      `UPDATE evaluation_executions SET status='completed',passed_count=$1,failed_count=$2,gate_passed=$3,summary=$4,completed_at=now()
       WHERE tenant_id=$5 AND id=$6`,
      [counts.passed ?? 0, (counts.failed ?? 0) + (counts.error ?? 0), gateFailures.length === 0, JSON.stringify(summary), tenantId, executionId],
    );
  }

  private thresholdFailures(thresholds: EvaluationThresholds, passRate: number, metrics: Record<string, number>): string[] {
    const failures: string[] = [];
    const minimums: Array<[keyof EvaluationThresholds, string]> = [
      ["minimumTaskSuccess", "task_success"], ["minimumToolSelection", "tool_selection"], ["minimumArgumentCorrectness", "argument_correctness"], ["minimumRetrievalQuality", "retrieval_quality"], ["minimumCitationCorrectness", "citation_correctness"], ["minimumApprovalCorrectness", "approval_correctness"], ["minimumSafety", "safety"], ["minimumRecoverySuccess", "recovery_success"],
    ];
    if (passRate < thresholds.minimumPassRate) failures.push("minimumPassRate");
    for (const [threshold, metric] of minimums) if ((metrics[metric] ?? 0) < Number(thresholds[threshold])) failures.push(String(threshold));
    const maximums: Array<[keyof EvaluationThresholds, string]> = [["maximumAverageLatencyMs", "latency_ms"], ["maximumAverageTokens", "tokens"], ["maximumAverageCostMicrousd", "cost_microusd"], ["maximumAverageSteps", "steps"]];
    for (const [threshold, metric] of maximums) { const limit = thresholds[threshold]; if (limit !== null && (metrics[metric] ?? 0) > Number(limit)) failures.push(String(threshold)); }
    return failures;
  }

  private candidateSummary(row: Row): Record<string, unknown> {
    const snapshot = row.agent_snapshot as { name?: string; systemInstructions?: string; model?: { provider?: string; model?: string } };
    return { label: row.candidate_label, agentId: row.agent_id, agentVersion: Number(row.agent_version), snapshotHash: row.agent_snapshot_hash, promptHash: canonicalJsonHash(snapshot.systemInstructions ?? ""), provider: snapshot.model?.provider, model: snapshot.model?.model, gatePassed: row.gate_passed };
  }
}
