import type pg from "pg";
import { resolve } from "node:path";
import { z } from "zod";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import type { Principal } from "../db/types.js";
import type { ArtifactStore } from "./backend.js";
import { MlBackendFailure } from "./contract.js";
import { TrainingSnapshot, type ArtifactReference, type Row, type Snapshot } from "./domain.js";
import { getModel } from "./model-catalog.js";
import { audit, authorizeMl, transaction } from "./persistence.js";
import { callRunner } from "./runner-process.js";
import { validatePredictions } from "./tasks.js";

export interface OnnxConfig { python: string; timeoutMs: number }
/** Parity rows: the attempt's held-out test partition, at most this many rows. */
export const PARITY_ROWS = 256;
/**
 * The tolerance policy every export is held to (and stored with). ONNX inference runs in float32 while the native
 * scikit-learn pipeline runs in float64, so values are compared with an absolute-plus-relative bound; class labels must
 * agree on every row.
 */
export const PARITY_TOLERANCE = { regression: { absolute: 1e-4, relative: 1e-3 }, classification: { labelAgreement: 1, probabilityAbsolute: 1e-4 } } as const;
const Outputs = z.object({ predictions: z.array(z.union([z.number().finite(), z.string(), z.boolean()])), probabilities: z.array(z.array(z.number().finite())).nullable() });
const Exported = z.object({
  onnx: z.string().min(1), signature: z.object({ inputs: z.array(z.object({ name: z.string(), elemType: z.string(), shape: z.array(z.int().nullable()) })).length(1), outputs: z.array(z.object({ name: z.string(), elemType: z.string() })) }),
  opsets: z.record(z.string(), z.int()), operators: z.array(z.string()).max(100), irVersion: z.int(), native: Outputs, classes: z.array(z.union([z.number(), z.string()])).nullable(),
  producer: z.record(z.string(), z.string()),
});
const Predicted = Outputs.extend({ runtime: z.object({ onnxruntime: z.string() }) });
type Parity = ReturnType<typeof parity> & { runtime?: { onnxruntime: string } };

/** Compares ONNX Runtime output with the native artifact's output on the same rows under PARITY_TOLERANCE. */
export function parity(task: "classification" | "regression", native: z.infer<typeof Outputs>, onnx: z.infer<typeof Outputs>) {
  const rows = native.predictions.length, sameLength = onnx.predictions.length === rows;
  if (task === "regression") {
    const t = PARITY_TOLERANCE.regression; let maxAbsolute = 0, maxRelative = 0, within = sameLength;
    for (let i = 0; sameLength && i < rows; i++) {
      const n = Number(native.predictions[i]), o = Number(onnx.predictions[i]), error = Math.abs(o - n);
      maxAbsolute = Math.max(maxAbsolute, error); maxRelative = Math.max(maxRelative, error / Math.max(Math.abs(n), Number.EPSILON));
      if (!(error <= t.absolute + t.relative * Math.abs(n))) within = false;
    }
    return { task, rows, tolerance: t, maxAbsoluteError: maxAbsolute, maxRelativeError: maxRelative, labelAgreement: null, maxProbabilityError: null, passed: within };
  }
  const t = PARITY_TOLERANCE.classification;
  const agreement = sameLength && rows ? native.predictions.filter((label, i) => label === onnx.predictions[i]).length / rows : 0;
  let probability: number | null = null;
  if (native.probabilities && onnx.probabilities) {
    probability = 0;
    for (const [i, row] of native.probabilities.entries()) {
      const other = onnx.probabilities[i];
      if (!other || other.length !== row.length) { probability = Infinity; break; }
      for (const [j, p] of row.entries()) probability = Math.max(probability, Math.abs(p - other[j]!));
    }
  }
  return { task, rows, tolerance: t, maxAbsoluteError: null, maxRelativeError: null, labelAgreement: agreement, maxProbabilityError: probability, passed: agreement >= t.labelAgreement && (probability === null || probability <= t.probabilityAbsolute) };
}

/**
 * ONNX export of registered model versions and ONNX Runtime inference. Only combinations the runner implements are
 * exported (scikit-learn classification and regression over numeric features); every other request is recorded as
 * UNSUPPORTED with its reason. An export is VERIFIED only if its runtime output matches the native artifact's.
 */
export class MlOnnxService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly config: OnnxConfig, private readonly script = resolve(process.cwd(), "ml/onnx_runner.py")) {}
  private call(request: Record<string, unknown>, signal: AbortSignal) { return callRunner({ label: "onnx", python: this.config.python, script: this.script, timeoutMs: this.config.timeoutMs, signal, request }); }

  async export(principal: Principal, modelVersionId: string, signal: AbortSignal) {
    authorizeMl(principal, true);
    const m = (await this.pool.query(`SELECT m.id,m.status,m.version,m.registry_entry_id,t.id AS training_run_id,t.snapshot,t.environment,t.split_indices,a.content,a.format
      FROM ml_model_versions m JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id
      WHERE m.tenant_id=$1 AND m.id=$2`, [principal.tenantId, modelVersionId])).rows[0];
    if (!m) throw new NotFoundError("Model version not found");
    if (m.status !== "REGISTERED" && m.status !== "READY") throw new ConflictError("Only REGISTERED or READY model versions can be exported");
    const snapshot = TrainingSnapshot.parse(m.snapshot), task = getModel(snapshot.backend, snapshot.algorithm).taskType;
    let record: { status: string; content?: ArtifactReference; described?: z.infer<typeof Exported>; parity?: Parity; failure?: { category: string; message: string } };
    try {
      if (task !== "classification" && task !== "regression") throw new MlBackendFailure("unsupported_capability", "Only classification and regression models can be exported to ONNX");
      const rows = await this.parityRows(principal.tenantId, snapshot, m.split_indices.test ?? []);
      const artifact = await this.artifacts.get(principal.tenantId, m.content);
      const exported = Exported.safeParse(await this.call({ command: "export", backend: snapshot.backend, snapshot, rows, artifact: artifact.toString("base64"), expectedEnvironment: m.environment,
        metadata: { "meticule.model_version_id": m.id, "meticule.training_run_id": m.training_run_id, "meticule.native_sha256": m.content.sha256, "meticule.native_format": m.format } }, signal));
      if (!exported.success) throw new MlBackendFailure("invalid_response", "ONNX runner response did not match the contract");
      const bytes = Buffer.from(exported.data.onnx, "base64");
      const runtime = await this.run(bytes, snapshot, rows, task, signal);
      const check = parity(task, exported.data.native, runtime);
      record = { status: check.passed ? "VERIFIED" : "PARITY_FAILED", content: await this.artifacts.put(principal.tenantId, bytes, "application/onnx"), described: exported.data, parity: { ...check, runtime: runtime.runtime } };
    } catch (error) {
      if (!(error instanceof MlBackendFailure)) throw error;
      // Only refusals that say the model or combination is unsupported are an outcome; an unavailable runner is an error.
      // Public storage: an export over the 16 MB blob limit cannot be kept either, so it is recorded the same way.
      if (error.category !== "unsupported_capability" && error.category !== "incompatible_artifact" && error.category !== "output_limit_exceeded") throw error;
      record = { status: "UNSUPPORTED", failure: { category: error.category, message: error.message.slice(0, 300) } };
    }
    return transaction(this.pool, async c => {
      const d = record.described;
      const r = await c.query("INSERT INTO ml_onnx_exports(tenant_id,model_version_id,status,content,opsets,signature,operators,producer,parity,failure,created_by) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING *",
        [principal.tenantId, modelVersionId, record.status, record.content ? JSON.stringify(record.content) : null, d ? JSON.stringify(d.opsets) : null, d ? JSON.stringify(d.signature) : null, d ? JSON.stringify(d.operators) : null,
          d ? JSON.stringify({ ...d.producer, irVersion: String(d.irVersion), classes: d.classes }) : null, record.parity ? JSON.stringify(record.parity) : null, record.failure ? JSON.stringify(record.failure) : null, principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.onnx_exported", { modelVersionId, exportId: r.rows[0].id, status: record.status });
      return r.rows[0];
    });
  }

  async list(principal: Principal, modelVersionId: string) {
    authorizeMl(principal);
    return (await this.pool.query("SELECT * FROM ml_onnx_exports WHERE tenant_id=$1 AND model_version_id=$2 ORDER BY created_at DESC,id LIMIT 100", [principal.tenantId, modelVersionId])).rows;
  }

  /** The ONNX bytes of an export, re-verified against their checksum. Managers only: it is the model itself. */
  async content(principal: Principal, exportId: string) {
    authorizeMl(principal, true);
    const e = (await this.pool.query("SELECT content FROM ml_onnx_exports WHERE tenant_id=$1 AND id=$2 AND content IS NOT NULL", [principal.tenantId, exportId])).rows[0];
    if (!e) throw new NotFoundError("ONNX export not found");
    return this.artifacts.get(principal.tenantId, e.content);
  }

  /** ONNX Runtime inference on validated rows; the output is held to the task's prediction contract like native output. */
  async predict(tenantId: string, content: ArtifactReference, snapshot: Snapshot, rows: Row[], signal: AbortSignal) {
    const task = getModel(snapshot.backend, snapshot.algorithm).taskType;
    if (task !== "classification" && task !== "regression") throw new MlBackendFailure("unsupported_capability", "ONNX inference supports classification and regression only");
    const output = await this.run(await this.artifacts.get(tenantId, content), snapshot, rows, task, signal);
    return validatePredictions(task, output.predictions, rows.length, {});
  }

  private async run(bytes: Buffer, snapshot: Snapshot, rows: Row[], task: "classification" | "regression", signal: AbortSignal) {
    const parsed = Predicted.safeParse(await this.call({ command: "predict", onnx: bytes.toString("base64"), features: snapshot.pipeline.definition.features, rows, task }, signal));
    if (!parsed.success) throw new MlBackendFailure("invalid_response", "ONNX runner response did not match the contract");
    return parsed.data;
  }

  /** Feature cells of the attempt's test rows (the target is never sent). */
  private async parityRows(tenantId: string, snapshot: Snapshot, test: number[]): Promise<Row[]> {
    const data = JSON.parse((await this.artifacts.get(tenantId, snapshot.dataset.content)).toString("utf8")) as Row[];
    const features = snapshot.pipeline.definition.features;
    const rows = test.slice(0, PARITY_ROWS).map(i => Object.fromEntries(features.map(f => [f, data[i]![f] ?? null])) as Row);
    if (!rows.length) throw new MlBackendFailure("unsupported_capability", "The training run has no test rows to verify an export against");
    return rows;
  }
}
