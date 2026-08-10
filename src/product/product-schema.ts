import { z } from "zod";

export const CreateMembershipSchema = z.object({
  identityId: z.uuid(), identityType: z.enum(["user", "service"]).default("user"),
  roles: z.array(z.string().trim().min(1).max(120)).max(50).refine((roles) => new Set(roles).size === roles.length, "Roles must be unique"),
});
export const AuditQuerySchema = z.object({ limit: z.coerce.number().int().min(1).max(200).default(100) });
export const RunListQuerySchema = z.object({ status: z.string().min(1).max(50).optional(), limit: z.coerce.number().int().min(1).max(200).default(100) });
