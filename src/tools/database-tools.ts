import { z } from "zod";
import type { ToolDefinition } from "./types.js";

export const noteStorageTool: ToolDefinition = {
  name: "structured_note_storage", description: "Persist a tenant-owned structured note.",
  inputSchema: z.object({ title: z.string().trim().min(1).max(200), body: z.string().min(1).max(10_000), tags: z.array(z.string().min(1).max(50)).max(20).default([]) }),
  outputSchema: z.object({ id: z.uuid(), createdAt: z.string() }), riskLevel: "medium",
  authorization: { requiredRoles: ["note_writer"] }, approvalRequirement: "policy", timeoutMs: 2000,
  retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "keyed_side_effect",
  retrySafety: "externally_idempotent",
  async execute(input, context) {
    const value = input as { title: string; body: string; tags: string[] };
    const result = await context.pool.query(
      `INSERT INTO structured_notes(tenant_id,created_by,title,body,tags,idempotency_key) VALUES($1,$2,$3,$4,$5,$6)
       ON CONFLICT(tenant_id,idempotency_key) DO UPDATE SET idempotency_key=excluded.idempotency_key RETURNING id,created_at`,
      [context.principal.tenantId, context.principal.userId, value.title, value.body, JSON.stringify(value.tags), context.idempotencyKey],
    );
    return { id: result.rows[0].id as string, createdAt: (result.rows[0].created_at as Date).toISOString() };
  },
};

export const readOnlySqlTool: ToolDefinition = {
  name: "demo_readonly_sql", description: "Run one read-only SELECT against the seeded demo catalog.",
  inputSchema: z.object({ query: z.string().trim().min(1).max(2000) }),
  outputSchema: z.object({ columns: z.array(z.string()), rows: z.array(z.record(z.string(), z.unknown())).max(100) }), riskLevel: "medium",
  authorization: { requiredRoles: ["demo_reader"] }, approvalRequirement: "never", timeoutMs: 2000,
  retryPolicy: { maxAttempts: 2, retryableErrors: ["40001"] }, idempotency: "pure",
  retrySafety: "pure",
  async execute(input, context) {
    const query = (input as { query: string }).query;
    if (!/^select\b/i.test(query) || /;|--|\/\*/.test(query)) throw new Error("Only one comment-free SELECT is allowed");
    const client = await context.pool.connect();
    try {
      await client.query("BEGIN READ ONLY"); await client.query("SET LOCAL ROLE agent_demo_reader"); await client.query("SET LOCAL statement_timeout='1500ms'");
      const result = await client.query({ text: query, rowMode: "array" });
      if (result.rows.length > 100) throw new Error("Query returned more than 100 rows");
      const columns = result.fields.map((field) => field.name);
      const rows = result.rows.map((values: unknown[]) => Object.fromEntries(columns.map((column, index) => [column, values[index]])));
      await client.query("ROLLBACK"); return { columns, rows };
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  },
};
