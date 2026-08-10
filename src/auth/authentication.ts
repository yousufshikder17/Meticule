import { jwtVerify, type JWTPayload } from "jose";
import { z } from "zod";
import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError } from "../domain/errors.js";

export class AuthenticationError extends Error { readonly status = 401; }
export interface Authenticator { authenticate(request: Request): Promise<Principal> }

const ClaimsSchema = z.object({
  sub: z.uuid(), tenant_id: z.uuid(), roles: z.array(z.string().min(1)).default([]), jti: z.string().min(1), identity_type: z.enum(["user", "service"]).default("user"),
});

export class JwtAuthenticator implements Authenticator {
  private readonly key: Uint8Array;
  constructor(private readonly pool: pg.Pool, secret: string, private readonly issuer: string, private readonly audience: string) { this.key = new TextEncoder().encode(secret); }
  async authenticate(request: Request): Promise<Principal> {
    const header = request.headers.get("authorization");
    if (!header?.startsWith("Bearer ")) throw new AuthenticationError("Bearer token is required");
    let payload: JWTPayload;
    try { ({ payload } = await jwtVerify(header.slice(7), this.key, { algorithms: ["HS256"], issuer: this.issuer, audience: this.audience })); }
    catch { throw new AuthenticationError("Token validation failed"); }
    const claims = ClaimsSchema.safeParse(payload);
    if (!claims.success) throw new AuthenticationError("Required token claims are invalid");
    const revoked = await this.pool.query("SELECT 1 FROM revoked_tokens WHERE issuer=$1 AND token_id=$2 AND expires_at>now()", [this.issuer, claims.data.jti]);
    if (revoked.rowCount) throw new AuthenticationError("Token has been revoked");
    const membership = await this.pool.query("SELECT roles,identity_type,status FROM tenant_memberships WHERE tenant_id=$1 AND identity_id=$2", [claims.data.tenant_id, claims.data.sub]);
    if (!membership.rowCount || membership.rows[0].status !== "active" || membership.rows[0].identity_type !== claims.data.identity_type) throw new AuthorizationError("Active tenant membership is required");
    const granted = new Set(membership.rows[0].roles as string[]);
    const roles = claims.data.roles.filter((role) => granted.has(role));
    return { tenantId: claims.data.tenant_id, userId: claims.data.sub, roles };
  }
}

export class DevelopmentHeaderAuthenticator implements Authenticator {
  async authenticate(request: Request): Promise<Principal> {
    const tenantId = z.uuid().safeParse(request.headers.get("x-tenant-id"));
    const userId = z.uuid().safeParse(request.headers.get("x-user-id"));
    if (!tenantId.success || !userId.success) throw new AuthenticationError("Development identity headers are required");
    const roles = (request.headers.get("x-roles") ?? "").split(",").map((role) => role.trim()).filter(Boolean);
    return { tenantId: tenantId.data, userId: userId.data, roles };
  }
}
