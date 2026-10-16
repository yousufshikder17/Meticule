import { spawnSync } from "node:child_process";
import { beforeAll, describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import type { MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import { ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2, MlBackendFailure, contractFor } from "../../src/ml/contract.js";
import type { ColumnType, Row, Snapshot } from "../../src/ml/domain.js";
import { getModel } from "../../src/ml/model-catalog.js";
import { TASKS, predictionCount } from "../../src/ml/tasks.js";
import { snapshotFor } from "../unit/ml-fixtures.js";

export type Column = { name: string; type: ColumnType; nullable?: boolean };

/**
 * Conformance for one (backend, task, model, fixture) of the Phase 4 modality extensions (spec section 10). The tabular suite
 * (contract-suite.ts) stays the contract for classification and regression over tabular inputs; this suite covers models
 * whose snapshots speak contract v2: non-tabular inputs, optional targets, task-specific outputs. Runs real runner processes.
 */
export interface TaskTarget {
  /** Backend registry id. */
  id: string;
  create(timeoutMs?: number): MlBackend;
  python: string;
  script: string;
  algorithm: string;
  hyperparameters: Record<string, unknown>;
  columns: Column[];
  /** The feature column (exactly one for the Phase 4 single-input models). */
  feature: string;
  /** The target column, or null for an unlabelled fixture. */
  target: string | null;
  rows: Row[];
  metrics: string[];
  /** Split override (e.g. chronological for forecasting); default random 60/20/20. */
  split?: Record<string, unknown>;
  /** Rows for predict (forecasting predicts from a history window); default the first four rows. */
  predictRows?: Row[];
  /** Fields the backend adds to every runner request (e.g. a model root); needed for the raw-envelope checks. */
  requestExtras?: Record<string, unknown>;
  requiredInCi: boolean;
}

const signal = () => new AbortController().signal;
const BUDGET = 300_000;

async function refusal(promise: Promise<unknown>): Promise<{ category: string; message: string }> {
  try { await promise; } catch (error) {
    if (error instanceof MlBackendFailure) return { category: error.category, message: error.message };
    if (error instanceof ConflictError) return { category: "invalid_request", message: error.message };
    throw new Error(`unclassified failure escaped the contract: ${error instanceof Error ? error.name : typeof error}`);
  }
  throw new Error("expected a refusal, but the call succeeded");
}
function expectBoundedMessage(message: string): void {
  expect(message).not.toMatch(/Traceback|File "|\.py\b|\\\\/);
  expect(message.length).toBeLessThan(500);
}
function stderrCode(stderr: string): string | undefined {
  try { return (JSON.parse(stderr.trim().split("\n").pop() ?? "") as { code?: string }).code; } catch { return undefined; }
}

export function describeTaskContract(target: TaskTarget): void {
  const definition = getModel(target.id, target.algorithm);
  const snapshot = (overrides: Parameters<typeof snapshotFor>[0] = {}): Snapshot => snapshotFor({
    backend: target.id, algorithm: target.algorithm, columns: target.columns, target: target.target, features: [target.feature], steps: [],
    hyperparameters: target.hyperparameters, rowCount: target.rows.length, evaluation: { metrics: target.metrics }, ...(target.split ? { split: target.split } : {}), ...overrides,
  });
  const predictRows = target.predictRows ?? target.rows.slice(0, 4);
  const task = TASKS[definition.taskType];
  const runRaw = (envelope: object) => {
    const result = spawnSync(target.python, [target.script], { input: JSON.stringify({ ...target.requestExtras, ...envelope }), encoding: "utf8", timeout: 120_000 });
    return { status: result.status, stderr: result.stderr ?? "" };
  };

  describe.runIf(Boolean(target.python))(`${target.id}/${target.algorithm} satisfies the ${definition.taskType} task contract (v2)`, () => {
    const backend = target.create();
    let prepared: PreparedData; let model: TrainedModel; let predictions: unknown[];
    beforeAll(async () => {
      const snap = snapshot();
      prepared = await backend.prepare(snap, target.rows, signal());
      model = await backend.train(snap, target.rows, prepared, signal());
      predictions = await backend.predict(snap, model.bytes, predictRows, model.environment, signal());
    }, BUDGET);

    if (target.requiredInCi) it("CI provides this runtime, so the suite cannot pass by skipping", () => { expect(target.python).toBeTruthy(); });

    describe("capability truthfulness", () => {
      it("declares the model's task and inputs, and its snapshots speak contract v2", () => {
        expect(backend.capabilities.tasks).toContain(definition.taskType);
        for (const type of definition.inputs.types) expect(backend.capabilities.inputs).toContain(type);
        expect(contractFor(snapshot(), definition.taskType)).toBe(ML_BACKEND_CONTRACT_V2);
      });
    });

    describe("validation refuses before any runner starts", () => {
      const offline = target.create(); // validation is synchronous: a refused snapshot never reaches a runner
      it("refuses a feature of another modality", () => {
        const wrong = target.columns.map(c => (c.name === target.feature ? { ...c, type: definition.inputs.types.includes("number") ? "text" as const : "number" as const } : c));
        expect(() => offline.validate(snapshot({ columns: wrong }))).toThrow(ConflictError);
      });
      it("refuses another backend's snapshot and the CV and split rules of the task", () => {
        expect(() => offline.validate({ ...snapshot(), backend: target.id === "sklearn" ? "pytorch" : "sklearn" } as Snapshot)).toThrow(ConflictError);
        const cv = snapshot({ evaluation: { metrics: target.metrics, cv: { strategy: "kfold", folds: 3 } } });
        // CV needs a task that allows it, a random split, and a model that fits something.
        if (!task.crossValidation || definition.learning === "pretrained" || target.split?.strategy === "chronological") expect(() => offline.validate(cv)).toThrow(ConflictError);
        else expect(() => offline.validate(cv)).not.toThrow();
      });
      it("applies the task's target rule", () => {
        if (target.target === null) return;
        // Default metrics: the labelled ones the fixture names would (rightly) be refused without a target.
        // Without a target there is nothing to stratify by.
        const split = target.split?.strategy === "random" ? { ...target.split, stratify: false } : target.split;
        const unlabelled = snapshot({ columns: target.columns.filter(c => c.name !== target.target), target: null, evaluation: {}, ...(split ? { split } : {}) });
        if (task.target === "required") expect(() => offline.validate(unlabelled)).toThrow(ConflictError);
        else expect(() => offline.validate(unlabelled)).not.toThrow();
      });
    });

    describe("lifecycle", () => {
      it("evaluates every partition with only the requested, finite metrics", async () => {
        const metrics = await backend.evaluate(snapshot(), target.rows, prepared, model, signal());
        expect(new Set(metrics.map(m => m.partition))).toEqual(new Set(["train", "validation", "test"]));
        expect(new Set(metrics.map(m => m.name))).toEqual(new Set(target.metrics));
        for (const m of metrics) expect(Number.isFinite(m.value), `${m.name}/${m.partition}`).toBe(true);
      }, BUDGET);
      it("predicts the task's number of outputs in its output schema", () => {
        expect(predictions).toHaveLength(predictionCount(definition, snapshot(), predictRows.length));
        if (definition.taskType === "embedding") {
          const dimension = Number(model.environment.embedding_dimension);
          for (const vector of predictions as number[][]) { expect(vector).toHaveLength(dimension); expect(vector.every(Number.isFinite)).toBe(true); }
        }
      });
      it("is deterministic: the same snapshot gives the same predictions from a fresh attempt", async () => {
        const again = await backend.train(snapshot(), target.rows, prepared, signal());
        expect(await backend.predict(snapshot(), again.bytes, predictRows, again.environment, signal())).toEqual(predictions);
      }, BUDGET);
    });

    describe("artifacts", () => {
      it("refuses a foreign runtime and corrupt bytes, and tolerates provenance-only drift", async () => {
        const runtime = Object.keys(model.environment).find(k => ["torch", "numpy", "scipy"].includes(k))!;
        expect((await refusal(backend.predict(snapshot(), model.bytes, predictRows, { ...model.environment, [runtime]: "0.0.1" }, signal()))).category).toBe("incompatible_artifact");
        expect((await refusal(backend.predict(snapshot(), Buffer.from("not an artifact"), predictRows, model.environment, signal()))).category).toBe("incompatible_artifact");
        expect(await backend.predict(snapshot(), model.bytes, predictRows, { ...model.environment, adapter_sha256: "0".repeat(64) }, signal())).toHaveLength(predictions.length);
      }, BUDGET);
    });

    describe("bounded failures and raw envelopes", () => {
      it("reports cancellation before start and a timeout, with bounded messages", async () => {
        const aborted = new AbortController(); aborted.abort();
        expect((await refusal(backend.prepare(snapshot(), target.rows, aborted.signal))).category).toBe("cancelled");
        const timed = await refusal(target.create(1).prepare(snapshot(), target.rows, signal()));
        expect(timed.category).toBe("timed_out"); expectBoundedMessage(timed.message);
      }, 60_000);
      it("refuses v0, v1 carrying extended semantics and a mismatched v2 task; accepts v2", () => {
        const snap = snapshot();
        expect(stderrCode(runRaw({ command: "prepare", contract: "meticule-ml-backend/v0" }).stderr)).toBe("unsupported_contract");
        expect(stderrCode(runRaw({ command: "prepare", contract: ML_BACKEND_CONTRACT, snapshot: snap, rows: target.rows }).stderr)).toBe("unsupported_contract");
        const other = definition.taskType === "embedding" ? "classification" : "embedding";
        expect(stderrCode(runRaw({ command: "prepare", contract: ML_BACKEND_CONTRACT_V2, task: other, snapshot: snap, rows: target.rows }).stderr)).toBe("unsupported_task");
        expect(runRaw({ command: "prepare", contract: ML_BACKEND_CONTRACT_V2, task: definition.taskType, snapshot: snap, rows: target.rows }).status).toBe(0);
      }, BUDGET);
    });
  });
}
