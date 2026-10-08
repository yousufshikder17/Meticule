import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import type { Evaluation, MlBackend, PreparedData, TrainedModel } from "./backend.js";
import { Metric, validateSnapshot, type Row, type Snapshot } from "./domain.js";
import { assertMetricsSupported, DEFAULT_METRICS } from "./metrics.js";
import { getModel, validateHyperparameters } from "./model-catalog.js";
import { validatePipeline } from "./preprocessing.js";

const Environment = z.record(z.string(), z.string());
const Indices = z.array(z.int().nonnegative());
const Prepared = z.object({ indices: z.object({ train: Indices, validation: Indices, test: Indices }), folds: z.array(Indices).optional(), environment: Environment });
const Performance = z.record(z.string(), z.number().finite().nonnegative());

export class SklearnBackend implements MlBackend {
  readonly id = "sklearn";
  readonly artifactFormats = ["sklearn-pickle-v1"] as const;
  constructor(private readonly python: string, private readonly timeoutMs = 120_000, private readonly script = resolve(process.cwd(), "ml/sklearn_runner.py")) {}
  validate(snapshot: Snapshot): void {
    validateSnapshot(snapshot);
    if (snapshot.backend !== this.id) throw new ConflictError("Backend mismatch");
    const model = getModel(this.id, snapshot.algorithm); validateHyperparameters(model, snapshot.hyperparameters);
    validatePipeline(snapshot);
    const { cv, metrics } = snapshot.evaluation;
    if ((snapshot.split.stratify || cv?.strategy === "stratified_kfold") && model.taskType === "regression") throw new ConflictError("Regression cannot stratify by target");
    assertMetricsSupported(model, metrics ?? DEFAULT_METRICS[model.taskType]);
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
    const output = z.object({ artifact: z.string().min(1), format: z.literal("sklearn-pickle-v1"), environment: Environment, resolvedHyperparameters: z.record(z.string(), z.unknown()), performance: Performance.optional() }).parse(await this.call("train", snapshot, rows, { prepared, expectedEnvironment: prepared.environment }, signal));
    return { bytes: Buffer.from(output.artifact, "base64"), format: output.format, environment: output.environment, resolvedHyperparameters: output.resolvedHyperparameters, performance: output.performance };
  }
  async evaluate(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal) { return (await this.evaluateDetailed(snapshot, rows, prepared, model, signal)).metrics; }
  async evaluateDetailed(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<Evaluation> {
    return z.object({ metrics: z.array(Metric.omit({ id: true })), performance: Performance.optional() }).parse(await this.call("evaluate", snapshot, rows, { prepared, artifact: model.bytes.toString("base64"), expectedEnvironment: model.environment }, signal));
  }
  async predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]> {
    return z.object({ predictions: z.array(z.json()) }).parse(await this.call("predict", snapshot, rows, { artifact: artifact.toString("base64"), expectedEnvironment: environment }, signal)).predictions;
  }
}
