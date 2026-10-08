import type pg from "pg";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { validateRows } from "./domain.js";
import { MlExperimentService } from "./experiments.js";
import { MlTrainingService } from "./persistence.js";

/** CV folds must exactly partition the training indices: the leakage boundary the backend is trusted to honour. */
function assertFolds(train: number[], folds: number[][] | undefined, expected: boolean, count?: number): void {
  if (!expected) { if (folds) throw new Error("Backend returned unexpected CV folds"); return; }
  const flat = (folds ?? []).flat(); const allowed = new Set(train);
  if (!folds || folds.length !== count || folds.some(f => !f.length) || flat.length !== allowed.size || new Set(flat).size !== flat.length || flat.some(i => !allowed.has(i))) throw new Error("Backend returned invalid CV folds");
}

export class TrainingProcessor {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: ReadonlyMap<string, MlBackend>) {}
  async execute(runId: string, workerId: string): Promise<void> {
    const service = new MlTrainingService(this.pool, this.artifacts);
    const controller = new AbortController(); const started = Date.now();
    let attemptId: string | undefined;
    // Check ownership while backend compute is in flight, so cancellation/lease loss kills it.
    const guard = setInterval(() => {
      if (!attemptId) return;
      void this.pool.query("SELECT 1 FROM runs r JOIN ml_training_jobs j ON j.run_id=r.id AND j.tenant_id=r.tenant_id JOIN ml_training_runs t ON t.job_id=j.id AND t.tenant_id=j.tenant_id WHERE r.id=$1 AND r.status='running' AND r.lease_owner=$2 AND r.lease_expires_at>now() AND r.cancellation_requested_at IS NULL AND t.id=$3 AND t.ended_at IS NULL", [runId, workerId, attemptId])
        .then(r => { if (!r.rowCount) controller.abort(); }).catch(() => controller.abort());
    }, 500);
    try {
      const attempt = await service.begin(runId, workerId); attemptId = attempt.id; const snapshot = attempt.snapshot;
      const backend = this.backends.get(snapshot.backend); if (!backend) throw new Error("Training backend is not configured"); backend.validate(snapshot);
      const data = await this.artifacts.get(attempt.tenantId, snapshot.dataset.content);
      const rows = validateRows(snapshot.dataset.schema, JSON.parse(data.toString("utf8")));
      if (rows.length !== snapshot.dataset.rowCount) throw new Error("Dataset row count mismatch");
      const prepared = await backend.prepare(snapshot, rows, controller.signal);
      const partitions = Object.values(prepared.indices); const all = partitions.flat();
      if (partitions.some(p => !p.length) || all.length !== rows.length || new Set(all).size !== rows.length || all.some(i => !Number.isInteger(i) || i < 0 || i >= rows.length)) throw new Error("Backend returned invalid split indices");
      assertFolds(prepared.indices.train, prepared.folds, Boolean(snapshot.evaluation.cv), snapshot.evaluation.cv?.folds);
      await service.phase(runId, workerId, attempt.id, "TRAINING", prepared);
      const model = await backend.train(snapshot, rows, prepared, controller.signal);
      await service.phase(runId, workerId, attempt.id, "EVALUATING", undefined, model);
      const { metrics, performance } = backend.evaluateDetailed ? await backend.evaluateDetailed(snapshot, rows, prepared, model, controller.signal) : { metrics: await backend.evaluate(snapshot, rows, prepared, model, controller.signal) };
      const required = snapshot.evaluation.cv ? ["train", "validation", "test", "cv", "cv_fold"] : ["train", "validation", "test"];
      if (!metrics.length || required.some(p => !metrics.some(m => m.partition === p)) || (!snapshot.evaluation.cv && metrics.some(m => m.partition.startsWith("cv")))) throw new Error("Backend omitted evaluation partitions");
      await service.phase(runId, workerId, attempt.id, "SAVING", undefined, undefined, performance);
      const content = await this.artifacts.put(attempt.tenantId, model.bytes, "application/octet-stream");
      await service.finish(runId, workerId, attempt.id, snapshot, metrics, content, model.format, Date.now() - started);
    } catch (error) {
      await service.fail(runId, workerId, attemptId, error instanceof Error ? error.message.slice(0, 500) : "Training failed");
    } finally {
      clearInterval(guard); controller.abort();
      // A terminal job may complete a benchmark/search/AutoML experiment; failure here must never affect the job.
      await new MlExperimentService(this.pool, this.artifacts, this.backends).onJobTerminal(runId);
    }
  }
}
