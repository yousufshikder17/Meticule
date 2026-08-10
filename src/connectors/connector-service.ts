import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import { canonicalJsonHash } from "../domain/canonical-json.js";
import { ToolTimeoutError } from "../tools/types.js";
import { CreateConnectorSchema, RevokeConnectorSchema, type McpCallInput } from "./connector-schema.js";
import type { ConnectorCredentialResolver } from "./credential-resolver.js";
import { compileUntrustedSchema, MCP_PROTOCOL_VERSION, McpSchemaError } from "./mcp-schemas.js";
import { McpHttpTransport, McpTransportError } from "./mcp-transport.js";

type Row = Record<string, unknown>;
export interface ConnectorRecord { id: string; tenantId: string; name: string; endpointUrl: string; credentialRef: string | null; allowedTools: string[]; requiredRoles: string[]; rateLimitPerMinute: number; protocolVersion: string; status: "active" | "degraded" | "revoked"; serverCapabilities: unknown; serverInfo: unknown; serverInstructions: string | null; lastHealthStatus: string | null; lastHealthAt: Date | null; lastDiscoveredAt: Date | null; createdBy: string; version: number }
export interface ConnectorCatalogueEntry { id: string; connectorId: string; connectorName: string; toolName: string; description: string | null; inputSchema: Record<string, unknown>; serverInstructions: string | null }
export class ConnectorConfigurationError extends Error {}
export class ConnectorCallError extends Error { constructor(public readonly code: string, message: string) { super(message); this.name = "ConnectorCallError"; } }

export class ConnectorService {
  constructor(private readonly pool: pg.Pool, private readonly transport: McpHttpTransport, private readonly credentials: ConnectorCredentialResolver) {}

  async create(principal: Principal, raw: unknown): Promise<ConnectorRecord> {
    this.manage(principal); const input = CreateConnectorSchema.parse(raw); await this.transportEndpointCheck(input.endpointUrl);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query(
        `INSERT INTO connectors(tenant_id,name,transport,endpoint_url,credential_ref,allowed_tools,required_roles,rate_limit_per_minute,created_by)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9) RETURNING *`,
        [principal.tenantId, input.name, input.transport, input.endpointUrl, input.credentialRef, JSON.stringify(input.allowedTools), JSON.stringify(input.requiredRoles), input.rateLimitPerMinute, principal.userId],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'connector.registered',$3)", [principal.tenantId, principal.userId, JSON.stringify({ connectorId: result.rows[0].id, name: input.name, origin: new URL(input.endpointUrl).origin, credentialConfigured: Boolean(input.credentialRef) })]);
      await client.query("COMMIT"); return this.fromRow(result.rows[0]);
    } catch (error) {
      await client.query("ROLLBACK").catch(() => undefined);
      if ((error as { code?: string }).code === "23505") throw new ConflictError("Connector name already exists in tenant");
      throw error;
    } finally { client.release(); }
  }

  async list(principal: Principal): Promise<ConnectorRecord[]> { this.inspect(principal); return (await this.pool.query("SELECT * FROM connectors WHERE tenant_id=$1 ORDER BY name,id", [principal.tenantId])).rows.map((row) => this.fromRow(row)); }
  async get(principal: Principal, id: string): Promise<ConnectorRecord> { this.inspect(principal); return this.connector(principal.tenantId, id, false); }
  async tools(principal: Principal, id: string): Promise<Row[]> {
    this.inspect(principal); await this.connector(principal.tenantId, id, false);
    return (await this.pool.query(
      "SELECT id,connector_id,name,title,description,input_schema,output_schema,annotations,schema_hash,enabled,discovered_at FROM connector_tools WHERE tenant_id=$1 AND connector_id=$2 ORDER BY name,id",
      [principal.tenantId, id],
    )).rows;
  }

  async discover(principal: Principal, id: string, signal = new AbortController().signal): Promise<{ connector: ConnectorRecord; tools: number }> {
    this.manage(principal); const connector = await this.connector(principal.tenantId, id, false); if (connector.status === "revoked") throw new ConflictError("Revoked connector cannot be discovered"); const credential = this.credentials.resolve(connector.credentialRef);
    try {
      const discovery = await this.transport.discover({ endpointUrl: connector.endpointUrl, credential }, signal);
      if (!discovery.capabilities.tools || typeof discovery.capabilities.tools !== "object") throw new ConflictError("MCP server did not advertise tool capability");
      const names = new Set<string>(); const allowed = new Set(connector.allowedTools); const selected = [];
      for (const tool of discovery.tools) {
        if (names.has(tool.name)) throw new McpSchemaError(`MCP server returned duplicate tool ${tool.name}`); names.add(tool.name);
        compileUntrustedSchema(tool.inputSchema, true); if (tool.outputSchema) compileUntrustedSchema(tool.outputSchema, false);
        if (tool.execution?.taskSupport === "required" && allowed.has(tool.name)) throw new McpSchemaError(`MCP tool ${tool.name} requires the unsupported tasks extension`);
        selected.push({ ...tool, enabled: allowed.has(tool.name) });
      }
      const missing = connector.allowedTools.filter((name) => !names.has(name)); if (missing.length) throw new ConflictError(`Allowlisted MCP tools were not discovered: ${missing.join(", ")}`);
      const client = await this.pool.connect();
      try {
        await client.query("BEGIN"); await client.query("DELETE FROM connector_tools WHERE tenant_id=$1 AND connector_id=$2", [principal.tenantId, id]);
        for (const tool of selected) await client.query(
          `INSERT INTO connector_tools(tenant_id,connector_id,name,title,description,input_schema,output_schema,annotations,schema_hash,enabled)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10)`,
          [principal.tenantId, id, tool.name, tool.title ?? null, tool.description ?? null, JSON.stringify(tool.inputSchema), tool.outputSchema ? JSON.stringify(tool.outputSchema) : null, JSON.stringify(tool.annotations ?? {}), canonicalJsonHash({ input: tool.inputSchema, output: tool.outputSchema ?? null }), tool.enabled],
        );
        const updated = await client.query(
          `UPDATE connectors SET status='active',protocol_version=$1,server_capabilities=$2,server_info=$3,server_instructions=$4,
           last_health_status='healthy',last_health_at=now(),last_discovered_at=now(),version=version+1,updated_at=now() WHERE tenant_id=$5 AND id=$6 RETURNING *`,
          [MCP_PROTOCOL_VERSION, JSON.stringify(discovery.capabilities), discovery.serverInfo === null ? null : JSON.stringify(discovery.serverInfo), discovery.instructions, principal.tenantId, id],
        );
        await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'connector.capabilities_discovered',$3)", [principal.tenantId, principal.userId, JSON.stringify({ connectorId: id, discoveredTools: selected.length, enabledTools: selected.filter((tool) => tool.enabled).length, protocolVersion: MCP_PROTOCOL_VERSION })]);
        await client.query("COMMIT"); return { connector: this.fromRow(updated.rows[0]), tools: selected.length };
      } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; } finally { client.release(); }
    } catch (error) {
      await this.pool.query("UPDATE connectors SET status='degraded',last_health_status='unhealthy',last_health_at=now(),updated_at=now() WHERE tenant_id=$1 AND id=$2 AND status<>'revoked'", [principal.tenantId, id]);
      await this.audit(principal.tenantId, null, "user", principal.userId, "connector.discovery_failed", { connectorId: id, code: this.errorCode(error) }); throw error;
    }
  }

  async health(principal: Principal, id: string, signal = new AbortController().signal): Promise<{ healthy: boolean; checkedAt: string }> {
    this.inspect(principal); const connector = await this.connector(principal.tenantId, id, false); if (connector.status === "revoked") throw new ConflictError("Revoked connector cannot be health checked"); let healthy = false;
    try { const result = await this.transport.discover({ endpointUrl: connector.endpointUrl, credential: this.credentials.resolve(connector.credentialRef) }, signal); healthy = result.supportedVersions.includes(MCP_PROTOCOL_VERSION); }
    catch { healthy = false; }
    await this.pool.query("UPDATE connectors SET status=CASE WHEN $1 THEN 'active' ELSE 'degraded' END,last_health_status=CASE WHEN $1 THEN 'healthy' ELSE 'unhealthy' END,last_health_at=now(),updated_at=now() WHERE tenant_id=$2 AND id=$3", [healthy, principal.tenantId, id]);
    await this.audit(principal.tenantId, null, "user", principal.userId, "connector.health_checked", { connectorId: id, healthy }); return { healthy, checkedAt: new Date().toISOString() };
  }

  async revoke(principal: Principal, id: string, raw: unknown): Promise<ConnectorRecord> {
    this.manage(principal); const { reason } = RevokeConnectorSchema.parse(raw);
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const result = await client.query(
        `UPDATE connectors SET status='revoked',revoked_at=COALESCE(revoked_at,now()),revoked_by=COALESCE(revoked_by,$1),revocation_reason=COALESCE(revocation_reason,$2),version=version+1,updated_at=now()
         WHERE tenant_id=$3 AND id=$4 RETURNING *`, [principal.userId, reason, principal.tenantId, id],
      );
      if (!result.rowCount) throw new NotFoundError("Connector not found");
      await client.query("UPDATE connector_tools SET enabled=false WHERE tenant_id=$1 AND connector_id=$2", [principal.tenantId, id]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'connector.revoked',$3)", [principal.tenantId, principal.userId, JSON.stringify({ connectorId: id, reason })]);
      await client.query("COMMIT"); return this.fromRow(result.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async catalogue(tenantId: string, connectorIds: string[], maximumTools: number): Promise<ConnectorCatalogueEntry[]> {
    if (!connectorIds.length) return [];
    const connectors = await this.pool.query("SELECT id FROM connectors WHERE tenant_id=$1 AND id=ANY($2::uuid[]) AND status='active' AND last_discovered_at IS NOT NULL", [tenantId, connectorIds]);
    if (connectors.rowCount !== new Set(connectorIds).size) throw new ConflictError("Configured connector is missing, revoked, degraded, undiscovered, or outside the tenant");
    const result = await this.pool.query(
      `SELECT ct.id,ct.connector_id,c.name AS connector_name,ct.name AS tool_name,ct.description,ct.input_schema,c.server_instructions
       FROM connector_tools ct JOIN connectors c ON c.tenant_id=ct.tenant_id AND c.id=ct.connector_id
       WHERE ct.tenant_id=$1 AND ct.connector_id=ANY($2::uuid[]) AND ct.enabled AND c.status='active' ORDER BY c.name,ct.name LIMIT $3`,
      [tenantId, connectorIds, maximumTools],
    );
    return result.rows.map((row) => ({ id: row.id, connectorId: row.connector_id, connectorName: row.connector_name, toolName: row.tool_name, description: row.description, inputSchema: row.input_schema, serverInstructions: row.server_instructions }));
  }

  async authorizeCall(context: { principal: Principal; runId: string }, input: McpCallInput): Promise<void> { await this.prepareCall(context, input); }

  async call(context: { principal: Principal; runId: string; idempotencyKey: string; signal: AbortSignal }, input: McpCallInput): Promise<unknown> {
    const prepared = await this.prepareCall(context, input); const requestId = crypto.randomUUID(); const execution = await this.pool.query("SELECT id FROM tool_executions WHERE tenant_id=$1 AND idempotency_key=$2", [context.principal.tenantId, context.idempotencyKey]);
    const invocation = await this.pool.query(
      `INSERT INTO connector_invocations(tenant_id,connector_id,run_id,tool_execution_id,idempotency_key,request_id,tool_name,status)
       VALUES($1,$2,$3,$4,$5,$6,$7,'pending') RETURNING id`,
      [context.principal.tenantId, input.connectorId, context.runId, execution.rows[0]?.id ?? null, context.idempotencyKey, requestId, input.toolName],
    ).catch((error: unknown) => { if ((error as { code?: string }).code === "23505") throw new ConflictError("Connector invocation identity was already used"); throw error; });
    const invocationId = invocation.rows[0].id as string; const started = Date.now();
    try {
      await this.consumeRateLimit(context.principal.tenantId, prepared.connector);
      await this.pool.query("UPDATE connector_invocations SET status='running',started_at=now(),updated_at=now() WHERE id=$1", [invocationId]);
      await this.audit(context.principal.tenantId, context.runId, "worker", context.principal.userId, "connector.invocation_started", { connectorId: input.connectorId, invocationId, toolName: input.toolName, requestId });
      const result = await this.transport.callTool({ endpointUrl: prepared.connector.endpointUrl, credential: this.credentials.resolve(prepared.connector.credentialRef) }, input.toolName, input.arguments, context.signal);
      if (prepared.outputSchema) {
        if (result.structuredContent === undefined || !prepared.outputSchema(result.structuredContent)) throw new McpTransportError("invalid_tool_output", "MCP structured output did not match its discovered schema", true);
      }
      const output = { ...result, connectorId: input.connectorId, toolName: input.toolName };
      await this.pool.query("UPDATE connector_invocations SET status='succeeded',duration_ms=$1,result_metadata=$2,completed_at=now(),updated_at=now() WHERE id=$3", [Date.now() - started, JSON.stringify({ isError: result.isError, contentTypes: result.content.map((block) => block.type), structured: result.structuredContent !== undefined }), invocationId]);
      await this.audit(context.principal.tenantId, context.runId, "worker", context.principal.userId, "connector.invocation_succeeded", { connectorId: input.connectorId, invocationId, toolName: input.toolName, isError: result.isError }); return output;
    } catch (error) {
      const unknown = error instanceof McpTransportError && error.outcomeUnknown;
      await this.pool.query("UPDATE connector_invocations SET status=$1,duration_ms=$2,error_code=$3,completed_at=now(),updated_at=now() WHERE id=$4", [unknown ? "unknown" : "failed", Date.now() - started, this.errorCode(error), invocationId]);
      await this.audit(context.principal.tenantId, context.runId, "worker", context.principal.userId, unknown ? "connector.invocation_unknown" : "connector.invocation_failed", { connectorId: input.connectorId, invocationId, toolName: input.toolName, code: this.errorCode(error) });
      if (unknown) throw new ToolTimeoutError("MCP tool outcome is uncertain and must not be replayed"); throw error;
    }
  }

  private async prepareCall(context: { principal: Principal; runId: string }, input: McpCallInput): Promise<{ connector: ConnectorRecord; outputSchema: ((value: unknown) => boolean) | null }> {
    const run = await this.pool.query("SELECT agent_configuration_snapshot->'connectorPolicy' AS connector_policy FROM runs WHERE tenant_id=$1 AND id=$2", [context.principal.tenantId, context.runId]);
    if (!run.rowCount) throw new NotFoundError("Run not found"); const policy = run.rows[0].connector_policy as { enabled?: boolean; connectorIds?: string[] };
    if (!policy.enabled || !policy.connectorIds?.includes(input.connectorId)) throw new AuthorizationError("Connector is not enabled for this agent");
    const connector = await this.connector(context.principal.tenantId, input.connectorId, true);
    for (const role of connector.requiredRoles) if (!context.principal.roles.includes(role)) throw new AuthorizationError(`Missing connector role: ${role}`);
    if (!connector.allowedTools.includes(input.toolName)) throw new AuthorizationError("MCP tool is not allowlisted for this connector");
    const tool = await this.pool.query("SELECT input_schema,output_schema FROM connector_tools WHERE tenant_id=$1 AND connector_id=$2 AND name=$3 AND enabled", [context.principal.tenantId, input.connectorId, input.toolName]);
    if (!tool.rowCount) throw new AuthorizationError("MCP tool was not discovered or is disabled");
    const validateInput = compileUntrustedSchema(tool.rows[0].input_schema, true); if (!validateInput(input.arguments)) throw new ConnectorCallError("invalid_arguments", "MCP arguments do not match the discovered schema");
    return { connector, outputSchema: tool.rows[0].output_schema ? compileUntrustedSchema(tool.rows[0].output_schema, false) : null };
  }

  private async consumeRateLimit(tenantId: string, connector: ConnectorRecord): Promise<void> {
    const result = await this.pool.query(
      `INSERT INTO connector_rate_limits(tenant_id,connector_id,window_start,request_count) VALUES($1,$2,date_trunc('minute',now()),1)
       ON CONFLICT(tenant_id,connector_id,window_start) DO UPDATE SET request_count=connector_rate_limits.request_count+1
       WHERE connector_rate_limits.request_count < $3 RETURNING request_count`, [tenantId, connector.id, connector.rateLimitPerMinute],
    ); if (!result.rowCount) throw new ConnectorCallError("rate_limited", "Connector rate limit exceeded");
  }
  private async connector(tenantId: string, id: string, requireActive: boolean): Promise<ConnectorRecord> { const result = await this.pool.query(`SELECT * FROM connectors WHERE tenant_id=$1 AND id=$2 ${requireActive ? "AND status='active'" : ""}`, [tenantId, id]); if (!result.rowCount) throw new NotFoundError("Connector not found or unavailable"); return this.fromRow(result.rows[0]); }
  private async transportEndpointCheck(endpointUrl: string): Promise<void> { await this.transport.validateEndpoint(endpointUrl); }
  private manage(principal: Principal): void { if (!principal.roles.includes("connector_manager")) throw new AuthorizationError("Connector management requires connector_manager role"); }
  private inspect(principal: Principal): void { if (!principal.roles.includes("connector_manager") && !principal.roles.includes("connector_auditor")) throw new AuthorizationError("Connector inspection requires connector_manager or connector_auditor role"); }
  private async audit(tenantId: string, runId: string | null, actorType: string, actorId: string, eventType: string, details: unknown): Promise<void> { await this.pool.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,$2,$3,$4,$5,$6)", [tenantId, runId, actorType, actorId, eventType, JSON.stringify(details)]); }
  private errorCode(error: unknown): string { return error && typeof error === "object" && "code" in error && typeof (error as { code?: unknown }).code === "string" ? (error as { code: string }).code : error instanceof McpSchemaError ? "invalid_schema" : "connector_error"; }
  private fromRow(row: Row): ConnectorRecord { return { id: row.id as string, tenantId: row.tenant_id as string, name: row.name as string, endpointUrl: row.endpoint_url as string, credentialRef: row.credential_ref as string | null, allowedTools: row.allowed_tools as string[], requiredRoles: row.required_roles as string[], rateLimitPerMinute: Number(row.rate_limit_per_minute), protocolVersion: row.protocol_version as string, status: row.status as ConnectorRecord["status"], serverCapabilities: row.server_capabilities, serverInfo: row.server_info, serverInstructions: row.server_instructions as string | null, lastHealthStatus: row.last_health_status as string | null, lastHealthAt: row.last_health_at as Date | null, lastDiscoveredAt: row.last_discovered_at as Date | null, createdBy: row.created_by as string, version: Number(row.version) }; }
}
