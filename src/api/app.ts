import { Hono } from "hono";
import { bodyLimit } from "hono/body-limit";
import { ZodError } from "zod";
import type pg from "pg";
import { AgentRepository, RunRepository } from "../db/repositories.js";
import type { Principal } from "../db/types.js";
import { CreateAgentSchema, CreateRunSchema, PatchAgentSchema, ResumeRunSchema, UuidSchema, type AgentConfiguration } from "../domain/schemas.js";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import { InvalidTransitionError } from "../domain/run-state.js";
import { createToolRegistry } from "../tools/registry.js";
import { ApprovalService } from "../approvals/approval-service.js";
import { ApprovalDecisionSchema } from "../domain/schemas.js";
import { AuthorizationError } from "../domain/errors.js";
import { AuthenticationError, type Authenticator } from "../auth/authentication.js";
import { withTenantSession } from "../db/tenant-session.js";
import { MemoryService } from "../memory/memory-service.js";
import { CreateMemorySchema, CorrectMemorySchema, MemoryListQuerySchema } from "../memory/memory-schema.js";
import type { RetrievalService } from "../retrieval/retrieval-service.js";
import { EmbeddingConfigurationError, EmbeddingProviderError } from "../retrieval/embedding-provider.js";
import { DeleteDocumentSchema, IngestDocumentSchema, SearchRetrievalSchema } from "../retrieval/retrieval-schema.js";
import type { ConnectorService } from "../connectors/connector-service.js";
import { ConnectorCallError, ConnectorConfigurationError } from "../connectors/connector-service.js";
import { ConnectorNetworkPolicyError } from "../connectors/network-policy.js";
import { ConnectorCredentialError } from "../connectors/credential-resolver.js";
import { McpTransportError } from "../connectors/mcp-transport.js";
import { McpSchemaError } from "../connectors/mcp-schemas.js";
import { SkillService } from "../skills/skill-service.js";
import { CreateSkillSchema, RevokeSkillSchema } from "../skills/skill-schema.js";
import { CreateConnectorSchema, RevokeConnectorSchema } from "../connectors/connector-schema.js";
import { OrchestrationService } from "../orchestration/orchestration-service.js";
import { EvaluationService } from "../evaluation/evaluation-service.js";
import { CreateEvaluationExecutionSchema, CreateEvaluationSuiteSchema } from "../evaluation/evaluation-schema.js";
import type { ProviderRegistry } from "../models/provider-registry.js";
import { NullLogger, type StructuredLogger } from "../observability/logger.js";
import { OperationalService } from "../observability/operational-service.js";
import { randomUUID } from "node:crypto";
import { PRODUCT_CSS, PRODUCT_HTML, PRODUCT_JS } from "../product/product-ui.js";
import { ProductService } from "../product/product-service.js";
import { AuditQuerySchema, CreateMembershipSchema, RunListQuerySchema } from "../product/product-schema.js";

export interface ReadinessControl { isDraining(): boolean; checkMigrations(): Promise<{ ready: boolean; missing: string[] }> }

type Variables = { principal: Principal; database: pg.Pool; correlationId: string };

export function createApp(pool: pg.Pool, authenticator: Authenticator, retrieval: RetrievalService | null = null, connectors: ConnectorService | null = null, providers: ProviderRegistry | null = null, structuredLogger: StructuredLogger = new NullLogger(), readiness: ReadinessControl | null = null): Hono<{ Variables: Variables }> {
  const app = new Hono<{ Variables: Variables }>();
  const tools = createToolRegistry(pool, retrieval, connectors);
  const requireRetrieval = (): RetrievalService => { if (!retrieval) throw new EmbeddingConfigurationError("Retrieval is disabled because no real embedding provider is configured"); return retrieval; };
  const requireConnectors = (): ConnectorService => { if (!connectors) throw new ConnectorConfigurationError("Connectors are disabled because no egress allowlist is configured"); return connectors; };
  const operations = new OperationalService(pool, providers);

  app.use("*", async (c, next) => { c.header("x-content-type-options", "nosniff"); c.header("referrer-policy", "no-referrer"); c.header("x-frame-options", "DENY"); c.header("content-security-policy", "default-src 'self'; connect-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"); await next(); });

  app.use("*", async (c, next) => {
    const correlationId = randomUUID(); const started = Date.now(); c.set("correlationId", correlationId); c.header("x-correlation-id", correlationId);
    try { await next(); }
    finally { structuredLogger.log("info", "http.request_completed", { correlationId, method: c.req.method, path: c.req.path, status: c.res.status, latencyMs: Date.now() - started }); }
  });

  app.onError((error, c) => {
    if (error instanceof ZodError) return c.json({ error: "validation_error", issues: error.issues }, 400);
    if (error instanceof AuthenticationError) return c.json({ error: "unauthorized", message: error.message }, 401);
    if (error instanceof NotFoundError) return c.json({ error: "not_found", message: error.message }, 404);
    if (error instanceof AuthorizationError) return c.json({ error: "forbidden", message: error.message }, 403);
    if (error instanceof EmbeddingConfigurationError) return c.json({ error: "retrieval_not_configured", message: error.message }, 503);
    if (error instanceof EmbeddingProviderError) return c.json({ error: "embedding_provider_error", code: error.code, message: error.message }, error.code === "model_not_found" ? 422 : 502);
    if (error instanceof ConnectorConfigurationError || error instanceof ConnectorCredentialError) return c.json({ error: "connector_not_configured", message: error.message }, 503);
    if (error instanceof ConnectorNetworkPolicyError) return c.json({ error: "connector_network_denied", message: error.message }, 400);
    if (error instanceof McpSchemaError) return c.json({ error: "invalid_mcp_schema", message: error.message }, 422);
    if (error instanceof ConnectorCallError) return c.json({ error: error.code, message: error.message }, error.code === "rate_limited" ? 429 : 400);
    if (error instanceof McpTransportError) return c.json({ error: "mcp_transport_error", code: error.code, message: error.message }, 502);
    if (error instanceof ConflictError || error instanceof InvalidTransitionError) return c.json({ error: "conflict", message: error.message }, 409);
    structuredLogger.log("error", "http.request_failed", { correlationId: c.get("correlationId"), method: c.req.method, path: c.req.path, error });
    return c.json({ error: "internal_error" }, 500);
  });

  app.get("/health/live", (c) => c.json({ status: "ok" }));
  app.get("/health/ready", async (c) => { await pool.query("SELECT 1"); if (readiness?.isDraining()) return c.json({ status: "draining" }, 503); const migrations = await readiness?.checkMigrations(); if (migrations && !migrations.ready) return c.json({ status: "migrations_pending", missing: migrations.missing }, 503); return c.json({ status: "ok" }); });
  app.get("/", (c) => c.html(PRODUCT_HTML));
  app.get("/app.css", (c) => c.body(PRODUCT_CSS, 200, { "content-type": "text/css; charset=utf-8", "cache-control": "no-store" }));
  app.get("/app.js", (c) => c.body(PRODUCT_JS, 200, { "content-type": "text/javascript; charset=utf-8", "cache-control": "no-store" }));

  app.use("*", async (c, next) => {
    if (c.req.path.startsWith("/health/") || c.req.path === "/" || c.req.path === "/app.css" || c.req.path === "/app.js") return next();
    const principal = await authenticator.authenticate(c.req.raw);
    c.set("principal", principal);
    await withTenantSession(pool, principal.tenantId, async (database) => {
      c.set("database", database);
      await next();
    });
  });
  const documentBodyLimit = bodyLimit({ maxSize: 2_200_000, onError: (c) => c.json({ error: "payload_too_large", message: "Document request exceeds 2.2 MB" }, 413) });
  app.use("/documents", documentBodyLimit);
  app.use("/documents/*", documentBodyLimit);
  const configurationBodyLimit = bodyLimit({ maxSize: 120_000, onError: (c) => c.json({ error: "payload_too_large", message: "Configuration request exceeds 120 KB" }, 413) });
  app.use("/connectors", configurationBodyLimit); app.use("/connectors/*", configurationBodyLimit); app.use("/skills", configurationBodyLimit); app.use("/skills/*", configurationBodyLimit);
  app.use("/evaluation-suites", configurationBodyLimit); app.use("/evaluation-suites/*", configurationBodyLimit);

  app.get("/me", (c) => c.json(c.get("principal")));
  app.get("/providers", (c) => c.json((providers?.list() ?? []).map((id) => ({ id }))));
  app.get("/agents", async (c) => c.json(await new ProductService(c.get("database")).agents(c.get("principal"))));
  app.post("/agents", async (c) => c.json(await new AgentRepository(c.get("database")).create(c.get("principal"), CreateAgentSchema.parse(await c.req.json())), 201));
  app.get("/agents/:id", async (c) => c.json(await new AgentRepository(c.get("database")).get(c.get("principal").tenantId, UuidSchema.parse(c.req.param("id")))));
  app.patch("/agents/:id", async (c) => {
    const input = PatchAgentSchema.parse(await c.req.json());
    const { expectedVersion, ...rawPatch } = input;
    const patch = Object.fromEntries(Object.entries(rawPatch).filter(([, value]) => value !== undefined)) as Partial<AgentConfiguration>;
    return c.json(await new AgentRepository(c.get("database")).patch(c.get("principal"), UuidSchema.parse(c.req.param("id")), expectedVersion, patch));
  });
  app.post("/agents/:id/runs", async (c) => {
    const input = CreateRunSchema.parse(await c.req.json());
    return c.json(await new RunRepository(c.get("database")).create(c.get("principal"), UuidSchema.parse(c.req.param("id")), input.goal), 201);
  });
  app.get("/runs", async (c) => c.json(await new ProductService(c.get("database")).runs(c.get("principal"), RunListQuerySchema.parse(c.req.query()))));
  app.get("/runs/:id", async (c) => c.json(await new RunRepository(c.get("database")).get(c.get("principal").tenantId, UuidSchema.parse(c.req.param("id")))));
  app.post("/runs/:id/cancel", async (c) => c.json(await new RunRepository(c.get("database")).requestCancellation(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.post("/runs/:id/resume", async (c) => {
    const { expectedVersion } = ResumeRunSchema.parse(await c.req.json());
    return c.json(await new RunRepository(c.get("database")).transition(c.get("principal"), UuidSchema.parse(c.req.param("id")), expectedVersion, "queued"));
  });
  app.get("/runs/:id/steps", async (c) => c.json(await new RunRepository(c.get("database")).listSteps(c.get("principal").tenantId, UuidSchema.parse(c.req.param("id")))));
  app.get("/runs/:id/trace", async (c) => c.json(await new RunRepository(c.get("database")).trace(c.get("principal").tenantId, UuidSchema.parse(c.req.param("id")))));
  app.get("/operations/metrics", async (c) => c.json(await new OperationalService(c.get("database"), providers).metrics(c.get("principal"))));
  app.get("/operations/providers/health", async (c) => c.json(await operations.providerHealth(c.get("principal"), c.req.raw.signal)));
  app.get("/runs/:id/tree", async (c) => c.json(await new OrchestrationService(c.get("database")).tree(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.get("/runs/:id/approvals", async (c) => c.json(await new ApprovalService(c.get("database"), tools).listForRun(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.get("/approvals", async (c) => c.json(await new ProductService(c.get("database")).approvals(c.get("principal"))));
  app.get("/approvals/:id", async (c) => c.json(await new ApprovalService(c.get("database"), tools).get(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.post("/approvals/:id/approve", async (c) => {
    const input = ApprovalDecisionSchema.parse(await c.req.json().catch(() => ({})));
    return c.json(await new ApprovalService(c.get("database"), tools).decide(c.get("principal"), UuidSchema.parse(c.req.param("id")), "approved", input.comment));
  });
  app.post("/approvals/:id/reject", async (c) => {
    const input = ApprovalDecisionSchema.parse(await c.req.json().catch(() => ({})));
    return c.json(await new ApprovalService(c.get("database"), tools).decide(c.get("principal"), UuidSchema.parse(c.req.param("id")), "rejected", input.comment));
  });
  app.get("/tools", (c) => c.json(tools.list()));
  app.get("/usage", async (c) => c.json(await new ProductService(c.get("database")).usage(c.get("principal"))));
  app.get("/audit", async (c) => c.json(await new ProductService(c.get("database")).audit(c.get("principal"), AuditQuerySchema.parse(c.req.query()))));
  app.get("/memberships", async (c) => c.json(await new ProductService(c.get("database")).memberships(c.get("principal"))));
  app.post("/memberships", async (c) => c.json(await new ProductService(c.get("database")).addMembership(c.get("principal"), CreateMembershipSchema.parse(await c.req.json())), 201));
  app.post("/memberships/:id/revoke", async (c) => c.json(await new ProductService(c.get("database")).revokeMembership(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.get("/memories", async (c) => c.json(await new MemoryService(c.get("database")).list(c.get("principal"), MemoryListQuerySchema.parse(c.req.query()))));
  app.post("/memories", async (c) => c.json(await new MemoryService(c.get("database")).create(c.get("principal"), CreateMemorySchema.parse(await c.req.json())), 201));
  app.get("/memories/:id/provenance", async (c) => c.json(await new MemoryService(c.get("database")).provenance(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.post("/memories/:id/corrections", async (c) => c.json(await new MemoryService(c.get("database")).correct(c.get("principal"), UuidSchema.parse(c.req.param("id")), CorrectMemorySchema.parse(await c.req.json())), 201));
  app.delete("/memories/:id", async (c) => {
    const body = await c.req.json().catch(() => ({})) as { reason?: unknown };
    const reason = typeof body.reason === "string" ? body.reason.trim().slice(0, 2_000) : "user_requested_deletion";
    return c.json(await new MemoryService(c.get("database")).delete(c.get("principal"), UuidSchema.parse(c.req.param("id")), reason || "user_requested_deletion"));
  });
  app.get("/documents", async (c) => c.json(await requireRetrieval().list(c.get("principal"))));
  app.post("/documents", async (c) => c.json(await requireRetrieval().ingest(c.get("principal"), IngestDocumentSchema.parse(await c.req.json()), c.req.raw.signal), 201));
  app.post("/documents/:id/versions", async (c) => c.json(await requireRetrieval().ingest(c.get("principal"), IngestDocumentSchema.parse(await c.req.json()), c.req.raw.signal, UuidSchema.parse(c.req.param("id"))), 201));
  app.get("/documents/:id/versions", async (c) => c.json(await requireRetrieval().versions(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.delete("/documents/:id", async (c) => c.json({ deletedVersions: await requireRetrieval().delete(c.get("principal"), UuidSchema.parse(c.req.param("id")), DeleteDocumentSchema.parse(await c.req.json().catch(() => ({})))) }));
  app.post("/retrieval/search", async (c) => c.json(await requireRetrieval().search(c.get("principal"), SearchRetrievalSchema.parse(await c.req.json()), c.req.raw.signal)));
  app.get("/connectors", async (c) => c.json(await requireConnectors().list(c.get("principal"))));
  app.post("/connectors", async (c) => c.json(await requireConnectors().create(c.get("principal"), CreateConnectorSchema.parse(await c.req.json())), 201));
  app.get("/connectors/:id", async (c) => c.json(await requireConnectors().get(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.get("/connectors/:id/tools", async (c) => c.json(await requireConnectors().tools(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.post("/connectors/:id/discover", async (c) => c.json(await requireConnectors().discover(c.get("principal"), UuidSchema.parse(c.req.param("id")), c.req.raw.signal)));
  app.post("/connectors/:id/health", async (c) => c.json(await requireConnectors().health(c.get("principal"), UuidSchema.parse(c.req.param("id")), c.req.raw.signal)));
  app.post("/connectors/:id/revoke", async (c) => c.json(await requireConnectors().revoke(c.get("principal"), UuidSchema.parse(c.req.param("id")), RevokeConnectorSchema.parse(await c.req.json()))));
  app.get("/skills", async (c) => c.json(await new SkillService(c.get("database")).list(c.get("principal"))));
  app.post("/skills", async (c) => c.json(await new SkillService(c.get("database")).create(c.get("principal"), CreateSkillSchema.parse(await c.req.json())), 201));
  app.get("/skills/:id/versions", async (c) => c.json(await new SkillService(c.get("database")).versions(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.post("/skills/:id/versions", async (c) => c.json(await new SkillService(c.get("database")).create(c.get("principal"), CreateSkillSchema.parse(await c.req.json()), UuidSchema.parse(c.req.param("id"))), 201));
  app.post("/skills/:id/revoke", async (c) => c.json(await new SkillService(c.get("database")).revoke(c.get("principal"), UuidSchema.parse(c.req.param("id")), RevokeSkillSchema.parse(await c.req.json()))));
  app.get("/evaluation-suites", async (c) => c.json(await new EvaluationService(c.get("database")).listSuites(c.get("principal"))));
  app.post("/evaluation-suites", async (c) => c.json(await new EvaluationService(c.get("database")).createSuite(c.get("principal"), CreateEvaluationSuiteSchema.parse(await c.req.json())), 201));
  app.get("/evaluation-suites/:id", async (c) => c.json(await new EvaluationService(c.get("database")).getSuite(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.post("/evaluation-suites/:id/versions", async (c) => c.json(await new EvaluationService(c.get("database")).reviseSuite(c.get("principal"), UuidSchema.parse(c.req.param("id")), CreateEvaluationSuiteSchema.parse(await c.req.json())), 201));
  app.post("/evaluation-suites/:id/executions", async (c) => c.json(await new EvaluationService(c.get("database")).createExecution(c.get("principal"), UuidSchema.parse(c.req.param("id")), CreateEvaluationExecutionSchema.parse(await c.req.json())), 201));
  app.get("/evaluations", async (c) => c.json(await new EvaluationService(c.get("database")).listExecutions(c.get("principal"), c.req.query())));
  app.get("/evaluations/:id", async (c) => c.json(await new EvaluationService(c.get("database")).getExecution(c.get("principal"), UuidSchema.parse(c.req.param("id")))));
  app.get("/evaluations/:id/compare/:otherId", async (c) => c.json(await new EvaluationService(c.get("database")).compare(c.get("principal"), UuidSchema.parse(c.req.param("id")), UuidSchema.parse(c.req.param("otherId")))));

  return app;
}
