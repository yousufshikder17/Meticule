import { randomUUID } from "node:crypto";
import { SignJWT } from "jose";
import pg from "pg";
import { z } from "zod";

const EnvironmentSchema = z.object({
  DATABASE_URL: z.url(),
  JWT_SECRET: z.string().min(32),
  JWT_ISSUER: z.string().min(1).default("durable-agent-platform"),
  JWT_AUDIENCE: z.string().min(1).default("durable-agent-api"),
  DEV_TENANT_ID: z.uuid().default("11111111-1111-4111-8111-111111111111"),
  DEV_USER_ID: z.uuid().default("22222222-2222-4222-8222-222222222222"),
});

if (process.env.NODE_ENV === "production") throw new Error("Development identity bootstrap is prohibited in production");

const environment = EnvironmentSchema.parse(process.env);
const databaseUrl = new URL(environment.DATABASE_URL);
if (!["localhost", "127.0.0.1", "[::1]"].includes(databaseUrl.hostname)) {
  throw new Error("Development identity bootstrap accepts only a loopback PostgreSQL URL");
}

const roles = ["tenant_admin", "usage_viewer", "audit_viewer", "document_manager", "memory_manager", "skill_manager", "connector_manager", "connector_auditor", "evaluation_manager", "evaluation_viewer", "system_operator", "approval_reviewer"];
const pool = new pg.Pool({ connectionString: environment.DATABASE_URL });

try {
  await pool.query(
    `INSERT INTO tenant_memberships(tenant_id,identity_id,identity_type,roles,status)
     VALUES($1,$2,'user',$3,'active')
     ON CONFLICT(tenant_id,identity_id) DO UPDATE
     SET identity_type='user',roles=excluded.roles,status='active',updated_at=now()`,
    [environment.DEV_TENANT_ID, environment.DEV_USER_ID, JSON.stringify(roles)],
  );
  const token = await new SignJWT({ tenant_id: environment.DEV_TENANT_ID, identity_type: "user", roles })
    .setProtectedHeader({ alg: "HS256" })
    .setSubject(environment.DEV_USER_ID)
    .setIssuer(environment.JWT_ISSUER)
    .setAudience(environment.JWT_AUDIENCE)
    .setJti(randomUUID())
    .setIssuedAt()
    .setExpirationTime("8h")
    .sign(new TextEncoder().encode(environment.JWT_SECRET));
  console.error(`Provisioned local development identity ${environment.DEV_USER_ID} in tenant ${environment.DEV_TENANT_ID}; token expires in 8 hours.`);
  console.log(token);
} finally {
  await pool.end();
}
