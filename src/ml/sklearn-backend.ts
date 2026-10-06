import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import type { MlBackend, PreparedData, TrainedModel } from "./backend.js";
import { Metric, validateSnapshot, type Row, type Snapshot } from "./domain.js";

const commonForest = { n_estimators: z.int().min(1).max(1000).optional(), max_depth: z.int().min(1).max(100).nullable().optional(), min_samples_leaf: z.int().min(1).max(1000).optional() };
const parameters = new Map<string, z.ZodType>([
  ["ridge", z.strictObject({ alpha: z.number().min(0).max(1e6).optional() })],
  ["logistic_regression", z.strictObject({ C: z.number().gt(0).max(1e6).optional(), max_iter: z.int().min(1).max(10_000).optional() })],
  ["random_forest_classifier", z.strictObject(commonForest)], ["random_forest_regressor", z.strictObject(commonForest)],
]);
const Environment = z.record(z.string(), z.string());
const Prepared = z.object({ indices: z.object({ train: z.array(z.int().nonnegative()), validation: z.array(z.int().nonnegative()), test: z.array(z.int().nonnegative()) }), environment: Environment });

export class SklearnBackend implements MlBackend {
  readonly id = "sklearn";
  constructor(private readonly python: string, private readonly timeoutMs = 120_000, private readonly script = resolve(process.cwd(), "ml/sklearn_runner.py")) {}
  validate(snapshot: Snapshot): void {
    validateSnapshot(snapshot);
    if (snapshot.backend !== this.id) throw new ConflictError("Backend mismatch");
    const schema = parameters.get(snapshot.algorithm); if (!schema) throw new ConflictError("Unsupported scikit-learn algorithm"); schema.parse(snapshot.hyperparameters);
    if (snapshot.pipeline.definition.features.some(f => snapshot.dataset.schema.columns.find(c => c.name === f)?.type !== "number")) throw new ConflictError("Initial sklearn backend requires numeric features");
    for (const step of snapshot.pipeline.definition.steps) if (!["impute_median", "standard_scale"].includes(step.operation) || Object.keys(step.parameters).length) throw new ConflictError("Unsupported sklearn pipeline operation/parameters");
    if (snapshot.split.stratify && ["ridge", "random_forest_regressor"].includes(snapshot.algorithm)) throw new ConflictError("Regression cannot stratify by target");
  }
  private call(command: string, snapshot: Snapshot, rows: Row[], extra: object, signal: AbortSignal): Promise<unknown> {
    this.validate(snapshot); signal.throwIfAborted();
    return new Promise((resolveCall, reject) => {
      const child = spawn(this.python, [this.script], { windowsHide: true, shell: false, signal, env: { ...process.env, OMP_NUM_THREADS: "1", OPENBLAS_NUM_THREADS: "1", MKL_NUM_THREADS: "1" }, stdio: ["pipe", "pipe", "pipe"] });
      let size = 0; const chunks: Buffer[] = [];
      const timeout = setTimeout(() => child.kill(), this.timeoutMs);
      child.stdout.on("data", (chunk: Buffer) => { size += chunk.length; if (size > 64 * 1024 * 1024) child.kill(); else chunks.push(chunk); });
      child.stderr.resume();
      child.stdin.on("error", () => undefined);
      child.once("error", error => { clearTimeout(timeout); reject(error); });
      child.once("close", code => {
        clearTimeout(timeout);
        if (code !== 0 || size > 64 * 1024 * 1024) { reject(new Error(`sklearn ${command} failed (${code ?? "terminated"})`)); return; }
        try { resolveCall(JSON.parse(Buffer.concat(chunks).toString("utf8"))); } catch { reject(new Error("Invalid sklearn response")); }
      });
      child.stdin.end(JSON.stringify({ command, snapshot, rows, ...extra }));
    });
  }
  async prepare(snapshot: Snapshot, rows: Row[], signal: AbortSignal): Promise<PreparedData> { return Prepared.parse(await this.call("prepare", snapshot, rows, {}, signal)); }
  async train(snapshot: Snapshot, rows: Row[], prepared: PreparedData, signal: AbortSignal): Promise<TrainedModel> {
    const output = z.object({ artifact: z.string().min(1), format: z.literal("sklearn-pickle-v1"), environment: Environment, resolvedHyperparameters: z.record(z.string(), z.unknown()) }).parse(await this.call("train", snapshot, rows, { prepared, expectedEnvironment: prepared.environment }, signal));
    return { bytes: Buffer.from(output.artifact, "base64"), format: output.format, environment: output.environment, resolvedHyperparameters: output.resolvedHyperparameters };
  }
  async evaluate(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal) {
    const output = z.object({ metrics: z.array(Metric.omit({ id: true })) }).parse(await this.call("evaluate", snapshot, rows, { prepared, artifact: model.bytes.toString("base64"), expectedEnvironment: model.environment }, signal)); return output.metrics;
  }
  async predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]> {
    return z.object({ predictions: z.array(z.json()) }).parse(await this.call("predict", snapshot, rows, { artifact: artifact.toString("base64"), expectedEnvironment: environment }, signal)).predictions;
  }
}
