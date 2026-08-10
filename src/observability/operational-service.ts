import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError } from "../domain/errors.js";
import type { ProviderRegistry } from "../models/provider-registry.js";
import { redact } from "./redaction.js";

export class OperationalService {
  constructor(private readonly pool: pg.Pool | pg.PoolClient, private readonly providers: ProviderRegistry | null = null) {}

  private authorize(principal: Principal): void {
    if (!principal.roles.includes("system_operator")) throw new AuthorizationError("system_operator role is required");
  }

  async heartbeat(instanceId: string, kind: "worker" | "api", metadata: Record<string, unknown> = {}, draining = false): Promise<void> {
    await this.pool.query(`INSERT INTO operational_instances(instance_id,instance_kind,metadata,draining)
      VALUES($1,$2,$3,$4) ON CONFLICT(instance_id) DO UPDATE SET
      last_heartbeat_at=now(),draining=EXCLUDED.draining,metadata=EXCLUDED.metadata,version=operational_instances.version+1`,
    [instanceId, kind, JSON.stringify(metadata), draining]);
  }

  async metrics(principal: Principal): Promise<Record<string, unknown>> {
    this.authorize(principal);
    const queues = await this.pool.query("SELECT status,count(*)::int AS count FROM runs WHERE tenant_id=$1 GROUP BY status ORDER BY status", [principal.tenantId]);
    const leases = await this.pool.query(`SELECT count(*) FILTER (WHERE expires_at>now())::int AS active,
        count(*) FILTER (WHERE expires_at<=now())::int AS expired,
        coalesce(max(extract(epoch FROM (now()-heartbeat_at))),0)::float AS oldest_heartbeat_age_seconds
        FROM worker_leases WHERE tenant_id=$1`, [principal.tenantId]);
    const usage = await this.pool.query(`SELECT coalesce(sum(input_tokens),0)::bigint AS input_tokens,
        coalesce(sum(output_tokens),0)::bigint AS output_tokens,coalesce(sum(cost_microusd),0)::bigint AS cost_microusd
        FROM usage_records WHERE tenant_id=$1 AND created_at>=now()-interval '24 hours'`, [principal.tenantId]);
    const failures = await this.pool.query(`SELECT category,count(*)::int AS count FROM (
        SELECT coalesce(error_details->>'code','run_failed') AS category FROM runs WHERE tenant_id=$1 AND status='failed'
        UNION ALL SELECT coalesce(normalized_error_code,'provider_failed') FROM model_attempts WHERE tenant_id=$1 AND status='failed'
        UNION ALL SELECT coalesce(error_details->>'code','tool_failed') FROM tool_executions WHERE tenant_id=$1 AND status IN ('failed','unknown')
      ) failures GROUP BY category ORDER BY count DESC,category LIMIT 25`, [principal.tenantId]);
    const workers = await this.pool.query(`SELECT instance_id,instance_kind,started_at,last_heartbeat_at,draining,
        extract(epoch FROM (now()-last_heartbeat_at))::float AS heartbeat_age_seconds,
        last_heartbeat_at>now()-interval '90 seconds' AS healthy,metadata
        FROM operational_instances WHERE instance_kind='worker' ORDER BY instance_id`);
    return { generatedAt: new Date().toISOString(), queueDepth: Object.fromEntries(queues.rows.map((row) => [row.status, row.count])), leaseHealth: leases.rows[0], usageLast24Hours: usage.rows[0], failureCategories: failures.rows, workers: workers.rows };
  }

  async providerHealth(principal: Principal, signal: AbortSignal): Promise<Array<Record<string, unknown>>> {
    this.authorize(principal);
    if (!this.providers) return [];
    return Promise.all(this.providers.list().map(async (id) => {
      const started = Date.now();
      try { const result = await this.providers!.get(id).healthCheck(signal); return { providerId: id, healthy: result.healthy, latencyMs: Date.now() - started, details: redact(result.details ?? null) }; }
      catch (error) { return { providerId: id, healthy: false, latencyMs: Date.now() - started, details: redact(error instanceof Error ? error.message.slice(0, 500) : "health check failed") }; }
    }));
  }
}
