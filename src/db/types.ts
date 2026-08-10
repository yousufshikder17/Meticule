import type { PoolClient } from "pg";
import type { Run, AgentConfiguration } from "../domain/schemas.js";

export interface Principal { tenantId: string; userId: string; roles: string[] }
export interface AgentRecord extends AgentConfiguration {
  id: string; tenantId: string; createdBy: string; version: number; createdAt: Date; updatedAt: Date;
}
export interface ClaimedRun { run: Run; workerId: string }
export type DatabaseClient = Pick<PoolClient, "query">;
