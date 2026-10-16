import type pg from "pg";
import type { ArtifactStore, MlBackend, PreparedData, TrainedModel } from "./backend.js";
import { ConflictError } from "../domain/errors.js";
import { failureCategory, MlBackendFailure } from "./contract.js";
import { validateRows, type ArtifactReference, type Row, type Snapshot } from "./domain.js";
import { MlExperimentService } from "./experiments.js";
import { MlTrainingService } from "./persistence.js";
import { EarlyStopper, effectivePolicy, isIterative, requestedEpochs } from "./training-policy.js";
import { getModel } from "./model-catalog.js";
import { TASKS, embeddingDimension } from "./tasks.js";
import { timestampMs } from "./media.js";

/**
 * Only failures Meticule classified carry their own text. Any other error (filesystem, database driver, parser) is
 * replaced, because its message can name server paths or echo data, and stored failures are returned by the API.
 */
function safeFailureMessage(error: unknown): string {
  return error instanceof MlBackendFailure || error instanceof ConflictError ? error.message.slice(0, 500) : "Training failed; see worker logs";
}

/** CV folds must exactly partition the training indices: the leakage boundary the backend is trusted to honour. */
function assertFolds(train: number[], folds: number[][] | undefined, expected: boolean, count?: number): void {
  if (!expected) { if (folds) throw new MlBackendFailure("invalid_response", "Backend returned unexpected CV folds"); return; }
  const flat = (folds ?? []).flat(); const allowed = new Set(train);
  if (!folds || folds.length !== count || folds.some(f => !f.length) || flat.length !== allowed.size || new Set(flat).size !== flat.length || flat.some(i => !allowed.has(i))) throw new MlBackendFailure("invalid_response", "Backend returned invalid CV folds");
}

/**
 * A chronological split puts every training instant at or before every validation instant, and those at or before every
 * test instant (ties allowed). A missing or unparsable time is NaN, which fails both comparisons.
 */
function assertChronological(rows: Row[], column: string, prepared: PreparedData): void {
  const times = (indices: number[]) => indices.map(i => timestampMs(String(rows[i]![column])) ?? NaN);
  const latest = (t: number[]) => t.reduce((a, b) => Math.max(a, b), -Infinity), earliest = (t: number[]) => t.reduce((a, b) => Math.min(a, b), Infinity);
  const [train, validation, test] = [prepared.indices.train, prepared.indices.validation, prepared.indices.test].map(times) as [number[], number[], number[]];
  if (!(latest(train) <= earliest(validation) && latest(validation) <= earliest(test))) throw new MlBackendFailure("invalid_response", "Backend returned a non-chronological split");
}

/** A resumed attempt must see exactly the split its checkpoint was trained on. Any difference fails explicitly. */
function assertSameSplit(previous: Record<string, number[]> | null, prepared: PreparedData): void {
  if (!previous) return;
  const same = ["train", "validation", "test"].every(k => JSON.stringify(previous[k]) === JSON.stringify((prepared.indices as Record<string, number[]>)[k]));
  if (!same) throw new MlBackendFailure("incompatible_checkpoint", "Resumed split differs from the checkpointed split");
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
      const backend = this.backends.get(snapshot.backend); if (!backend) throw new MlBackendFailure("backend_unavailable", "Training backend is not configured"); backend.validate(snapshot);
      // The task comes from the catalog only (B1), and its CV rule holds whichever backend runs the attempt.
      const task = getModel(snapshot.backend, snapshot.algorithm).taskType;
      if (snapshot.evaluation.cv && (!TASKS[task].crossValidation || snapshot.split.strategy !== "random")) throw new ConflictError("Cross-validation is not supported for this task and split");
      const data = await this.artifacts.get(attempt.tenantId, snapshot.dataset.content);
      const rows = validateRows(snapshot.dataset.schema, JSON.parse(data.toString("utf8")));
      if (rows.length !== snapshot.dataset.rowCount) throw new MlBackendFailure("training_failed", "Dataset row count mismatch");
      const prepared = await backend.prepare(snapshot, rows, controller.signal);
      const partitions = Object.values(prepared.indices); const all = partitions.flat();
      if (partitions.some(p => !p.length) || all.length !== rows.length || new Set(all).size !== rows.length || all.some(i => !Number.isInteger(i) || i < 0 || i >= rows.length)) throw new MlBackendFailure("invalid_response", "Backend returned invalid split indices");
      if (snapshot.split.strategy === "chronological") assertChronological(rows, snapshot.split.timeColumn, prepared);
      assertFolds(prepared.indices.train, prepared.folds, Boolean(snapshot.evaluation.cv), snapshot.evaluation.cv?.folds);
      assertSameSplit(attempt.previousSplit, prepared);
      await service.phase(runId, workerId, attempt.id, "TRAINING", prepared);
      let model: TrainedModel; let outcome: { stopReason: "completed" | "early_stopping"; bestEpoch: number | null } | undefined;
      // Iterative means both: the backend can run chunks AND the catalog declares the model iterative.
      if (backend.capabilities.iterative && isIterative(snapshot)) {
        const trained = await this.trainIteratively(service, runId, workerId, attempt, backend, snapshot, rows, prepared, controller.signal);
        model = trained.model; outcome = trained.outcome;
      } else {
        model = await backend.train(snapshot, rows, prepared, controller.signal);
      }
      if (task === "embedding") embeddingDimension(model.environment); // E1: prediction checks vectors against it, so it must be valid
      await service.phase(runId, workerId, attempt.id, "EVALUATING", undefined, model);
      const { metrics, performance } = backend.evaluateDetailed ? await backend.evaluateDetailed(snapshot, rows, prepared, model, controller.signal) : { metrics: await backend.evaluate(snapshot, rows, prepared, model, controller.signal) };
      const required = snapshot.evaluation.cv ? ["train", "validation", "test", "cv", "cv_fold"] : ["train", "validation", "test"];
      if (!metrics.length || required.some(p => !metrics.some(m => m.partition === p)) || (!snapshot.evaluation.cv && metrics.some(m => m.partition.startsWith("cv")))) throw new MlBackendFailure("invalid_response", "Backend omitted evaluation partitions");
      await service.phase(runId, workerId, attempt.id, "SAVING", undefined, undefined, performance);
      // A backend may write only the artifact format it declares; anything else is a contract violation, never stored.
      if (!backend.capabilities.artifactFormats.includes(model.format)) throw new MlBackendFailure("invalid_response", "Backend wrote an undeclared artifact format");
      const content = await this.artifacts.put(attempt.tenantId, model.bytes, "application/octet-stream");
      await service.finish(runId, workerId, attempt.id, snapshot, metrics, content, model.format, Date.now() - started, outcome);
    } catch (error) {
      await service.fail(runId, workerId, attemptId, { category: failureCategory(error, "training_failed"), message: safeFailureMessage(error) });
    } finally {
      clearInterval(guard); controller.abort();
      // A terminal job may complete a benchmark/search/AutoML experiment; failure here must never affect the job.
      await new MlExperimentService(this.pool, this.artifacts, this.backends).onJobTerminal(runId);
    }
  }

  /**
   * Meticule owns the epoch loop. Each iteration runs one bounded backend chunk, then persists its summaries and
   * checkpoint under the lease fence. A lost lease or cancellation throws before anything is written.
   */
  private async trainIteratively(service: MlTrainingService, runId: string, workerId: string, attempt: { id: string; tenantId: string; jobId: string; resume: { epoch: number; content: ArtifactReference; format: string; schemaVersion: number } | null }, backend: MlBackend, snapshot: Snapshot, rows: Row[], prepared: PreparedData, signal: AbortSignal): Promise<{ model: TrainedModel; outcome: { stopReason: "completed" | "early_stopping"; bestEpoch: number | null } }> {
    if (!backend.trainChunk || !backend.exportModel) throw new MlBackendFailure("unsupported_capability", "Backend does not support iterative training");
    const policy = effectivePolicy(snapshot); const total = requestedEpochs(snapshot);
    let state: Buffer | null = null; let completed = 0;
    if (attempt.resume) {
      // Meticule refuses a checkpoint format the backend does not declare before any bytes are read or handed over.
      if (!backend.capabilities.checkpointFormats.includes(attempt.resume.format)) throw new MlBackendFailure("incompatible_checkpoint", "Checkpoint format is not one this backend can resume");
      // Corrupt, tampered or missing checkpoint bytes fail the store's content-hash check and are an invalid checkpoint.
      try { state = await this.artifacts.get(attempt.tenantId, attempt.resume.content); }
      catch { throw new MlBackendFailure("invalid_checkpoint", "Checkpoint bytes failed integrity verification"); }
      completed = attempt.resume.epoch;
    }
    const stopper = policy.earlyStopping
      ? EarlyStopper.replay(policy.earlyStopping, (await service.priorEpochs(attempt.tenantId, attempt.jobId)).filter(e => e.epoch <= completed))
      : null;
    let stopReason: "completed" | "early_stopping" = "completed";
    while (completed < total && !(stopper?.shouldStop)) {
      signal.throwIfAborted();
      const chunk = Math.min(policy.checkpointEveryEpochs, total - completed);
      const result = await backend.trainChunk(snapshot, rows, prepared, state, chunk, signal);
      const expected = Array.from({ length: chunk }, (_, i) => completed + i + 1);
      if (result.completedEpoch !== completed + chunk || result.epochs.map(e => e.epoch).join() !== expected.join()) throw new MlBackendFailure("invalid_response", "Backend returned unexpected epochs");
      const content = await this.artifacts.put(attempt.tenantId, result.checkpoint, "application/octet-stream");
      await service.checkpoint(runId, workerId, attempt.id, result.epochs, { epoch: result.completedEpoch, content, format: result.checkpointFormat, schemaVersion: result.checkpointSchema });
      state = result.checkpoint; completed = result.completedEpoch;
      if (stopper) {
        for (const e of result.epochs) stopper.observe(e);
        if (stopper.shouldStop) stopReason = "early_stopping";
      }
    }
    if (stopper?.shouldStop) stopReason = "early_stopping";
    const model = await backend.exportModel(snapshot, state!, Boolean(policy.earlyStopping), signal);
    return { model, outcome: { stopReason, bestEpoch: policy.earlyStopping ? stopper!.bestEpoch : null } };
  }
}
