import type pg from "pg";
import { RunRepository } from "../db/repositories.js";
import { NullLogger, type StructuredLogger } from "../observability/logger.js";

export interface WorkerOptions { workerId: string; leaseSeconds: number; kind?: "agent" | "ml" }
export interface RunProcessor { execute(runId: string, workerId: string): Promise<void> }

export class LifecycleWorker {
  private readonly runs: RunRepository;
  constructor(pool: pg.Pool, private readonly options: WorkerOptions, private readonly processor?: RunProcessor, private readonly logger: StructuredLogger = new NullLogger()) { this.runs = new RunRepository(pool); }

  async tick(): Promise<boolean> {
    await this.runs.recoverExpired();
    await this.runs.finalizeUnleasedCancellations();
    const claimed = await this.runs.claimNext(this.options.workerId, this.options.leaseSeconds, this.options.kind);
    if (!claimed) return false;
    const started = Date.now(); this.logger.log("info", "worker.run_claimed", { workerId: this.options.workerId, tenantId: claimed.tenantId, runId: claimed.id });
    await this.runs.workerTransition(claimed.id, this.options.workerId, "running");
    if (!this.processor) {
      await this.runs.workerTransition(claimed.id, this.options.workerId, "paused", { code: "executor_not_configured", message: "No execution processor is configured." });
      return true;
    }
    let leaseLost = false;
    const heartbeat = setInterval(() => {
      void this.runs.heartbeat(claimed.id, this.options.workerId, this.options.leaseSeconds)
        .then((owned) => { if (!owned) leaseLost = true; })
        .catch(() => { leaseLost = true; });
    }, Math.max(1000, Math.floor(this.options.leaseSeconds * 1000 / 3)));
    try { await this.processor.execute(claimed.id, this.options.workerId); }
    finally { clearInterval(heartbeat); }
    if (leaseLost) {
      const current = await this.runs.get(claimed.tenantId, claimed.id);
      const intentionallyReleased = !current.leaseOwner && ["waiting_for_approval", "paused", "completed", "failed", "cancelled"].includes(current.status);
      if (!intentionallyReleased) throw new Error("Worker lease was lost during execution");
    }
    this.logger.log("info", "worker.run_released", { workerId: this.options.workerId, tenantId: claimed.tenantId, runId: claimed.id, latencyMs: Date.now() - started });
    return true;
  }
}
