import { spawnSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import type { MlBackend, PreparedData, TrainedModel } from "../../src/ml/backend.js";
import { contractFor, ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2, MlBackendFailure } from "../../src/ml/contract.js";
import { type Row, type Snapshot, TrainingSnapshot } from "../../src/ml/domain.js";
import { getModel, listModels } from "../../src/ml/model-catalog.js";
import { OPERATIONS } from "../../src/ml/preprocessing.js";
import { snapshotFor } from "../unit/ml-fixtures.js";
// Model definitions for every internal backend, so "another backend" is always present. Importing them runs no framework code.
import "../../src/ml/pytorch-models.js";
import "../../src/ml/tensorflow-models.js";
import "../../src/ml/jax-models.js";

/**
 * Meticule ML Backend Contract v1 conformance. One suite, written once against the platform semantics, runs against every
 * internal backend. Each check is applied only where the backend's declared capabilities say it applies, and every
 * applicable check is asserted: none is skipped by a flag, and an unavailable runtime is tested through its refusal.
 *
 * Runs real runner processes. No database is needed: these checks exercise the backend contract, not the job lifecycle.
 */
export interface ConformanceTarget {
  /** Registry id, which must equal the backend's id. */
  id: string;
  /** Builds the backend with the given runner timeout. */
  create(timeoutMs?: number): MlBackend;
  /** Interpreter and runner script, for the raw protocol checks. */
  python: string;
  script: string;
  /** Small, fast configurations for one classification and one regression model. */
  classification: Configured;
  regression: Configured;
  /** True when CI must provide this runtime, so a missing interpreter fails the job instead of skipping. */
  requiredInCi: boolean;
}
interface Configured { algorithm: string; hyperparameters: Record<string, unknown> }

export const COLUMNS = [
  { name: "x", type: "number" as const },
  { name: "z", type: "number" as const },
  { name: "label", type: "string" as const },
  { name: "y", type: "number" as const },
];
export const ROWS: Row[] = Array.from({ length: 90 }, (_, i) => ({ x: i / 10, z: (i * 7 % 13) / 3, label: i % 3 === 0 ? "a" : "b", y: 2 * (i / 10) + (i * 7 % 13) / 3 }));
export const signal = () => new AbortController().signal;
const FRAMEWORK_TRAINING_BUDGET = 300_000;

// ---- Declared preprocessing: every operation a backend declares is executed and must behave as the platform defines it.
// Roles are keyed on the platform's own operation registry (src/ml/preprocessing.ts), not on any backend, and every
// registry operation must have one, so a new operation cannot be declared without a conformance expectation.
type Fill = (values: (number | string)[]) => number | string;
const median: Fill = values => { const s = (values as number[]).slice().sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2; };
const mean: Fill = values => (values as number[]).reduce((a, b) => a + b, 0) / values.length;
// Most frequent value; ties break by value, as the runners do. The fixture has no ties.
const mode: Fill = values => { const counts = new Map<string, number>(); for (const v of values as string[]) counts.set(v, (counts.get(v) ?? 0) + 1); return [...counts].sort((a, b) => b[1] - a[1] || (a[0] < b[0] ? -1 : 1))[0]![0]; };
export const OPERATION_EXPECTATIONS: Record<string, { role: "numeric_imputer" | "numeric_transform" | "categorical_encoder" | "categorical_imputer" | "feature_selector"; fill?: Fill }> = {
  impute_median: { role: "numeric_imputer", fill: median },
  impute_mean: { role: "numeric_imputer", fill: mean },
  standard_scale: { role: "numeric_transform" },
  min_max_scale: { role: "numeric_transform" },
  one_hot_encode: { role: "categorical_encoder" },
  ordinal_encode: { role: "categorical_encoder" },
  impute_most_frequent: { role: "categorical_imputer", fill: mode },
  variance_threshold: { role: "feature_selector" },
};
const PREP_COLUMNS = [
  { name: "x", type: "number" as const }, { name: "w", type: "number" as const }, { name: "xn", type: "number" as const, nullable: true },
  { name: "c", type: "number" as const }, { name: "kind", type: "string" as const }, { name: "kindn", type: "string" as const, nullable: true },
  { name: "y", type: "number" as const },
];
// xn is skewed (mean differs from median) and has missing values; kindn has a clear mode ("p") and missing values; c is constant.
const PREP_ROWS: Row[] = Array.from({ length: 90 }, (_, i) => {
  const xn = i % 10 === 0 ? null : ((i % 7) ** 2) / 4;
  const kind = ["a", "b", "c"][i % 3]!;
  const kindn = i % 10 === 5 ? null : i % 3 === 0 ? "q" : "p";
  return { x: i / 10, w: (i * 7 % 13) / 3, xn, c: 0, kind, kindn, y: i / 10 + 0.8 * (xn ?? 2) + (kind === "a" ? 1 : kind === "b" ? -1 : 0) + (kindn === "p" ? 0.7 : 0) };
});
const close = (a: number, b: number) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));

// ---- Declared scores: a binary and a three-class target that the features predict, so informative scores are measurable.
const SCORE_COLUMNS = [{ name: "x", type: "number" as const }, { name: "w", type: "number" as const }, { name: "bin", type: "string" as const }, { name: "tri", type: "string" as const }];
const SCORE_ROWS: Row[] = Array.from({ length: 90 }, (_, i) => {
  const x = i / 10;
  return { x, w: (i * 7 % 13) / 3, bin: x > 4.45 ? (i % 17 === 0 ? "no" : "yes") : (i % 19 === 0 ? "yes" : "no"), tri: x < 3 ? "a" : x < 6 ? "b" : "c" };
});

export function snapshotWith(backendId: string, configured: Configured, task: "classification" | "regression", overrides: Record<string, unknown> = {}): Snapshot {
  const base = snapshotFor({
    backend: backendId, algorithm: configured.algorithm, columns: COLUMNS, target: task === "classification" ? "label" : "y",
    features: ["x", "z"], steps: [{ operation: "standard_scale" }], hyperparameters: configured.hyperparameters, rowCount: ROWS.length,
  });
  return { ...base, ...overrides } as Snapshot;
}

// ---- Contract v2 on a registered tabular model: a non-feature timestamp column and a chronological split. Rows are out of
// time order (row i holds hour 37i mod 90), so a random split cannot pass for a chronological one.
export const CHRONO_ROWS: Row[] = ROWS.map((row, i) => ({ ...row, at: new Date(Date.UTC(2024, 0, 1) + (i * 37 % 90) * 3_600_000).toISOString() }));
export function chronologicalSnapshot(backendId: string, configured: Configured, task: "classification" | "regression"): Snapshot {
  return snapshotFor({
    backend: backendId, algorithm: configured.algorithm, columns: [...COLUMNS, { name: "at", type: "timestamp" }], target: task === "classification" ? "label" : "y",
    features: ["x", "z"], steps: [{ operation: "standard_scale" }], hyperparameters: configured.hyperparameters, rowCount: CHRONO_ROWS.length,
    split: { strategy: "chronological", timeColumn: "at" },
  });
}
/** Train, validation and test (each in its stored order, concatenated) are non-decreasing in time, with 54/18/18 rows. */
export function expectTimeOrdered(prepared: PreparedData): void {
  const { train, validation, test } = prepared.indices;
  expect([train.length, validation.length, test.length]).toEqual([54, 18, 18]);
  const times = [...train, ...validation, ...test].map(i => Date.parse(CHRONO_ROWS[i]!.at as string));
  for (let i = 1; i < times.length; i++) expect(times[i]!).toBeGreaterThanOrEqual(times[i - 1]!);
}

/** The same snapshot with a different preprocessing pipeline. The ids stay fixed, so only the pipeline changes. */
function withSteps(base: Snapshot, steps: { operation: string; parameters?: Record<string, unknown> }[]): Snapshot {
  return { ...base, pipeline: { ...base.pipeline, definition: { features: ["x", "z"], steps: steps.map(s => ({ operation: s.operation, parameters: s.parameters ?? {} })) } } } as Snapshot;
}

/** Parses the snapshot through the platform schema, so a policy that the schema refuses fails before the backend is asked. */
function withPolicy(snapshot: Snapshot, policy: Record<string, unknown>): Snapshot {
  return TrainingSnapshot.parse({ ...snapshot, training: { checkpointEveryEpochs: 2, earlyStopping: null, device: "cpu", precision: "fp32", ...policy } });
}

/** The category a refusal carries, or a failure that the contract did not classify (which fails the calling check). */
async function refusal(promise: Promise<unknown>): Promise<{ category: string; message: string }> {
  try { await promise; } catch (error) {
    if (error instanceof MlBackendFailure) return { category: error.category, message: error.message };
    if (error instanceof ConflictError) return { category: "invalid_request", message: error.message };
    throw new Error(`unclassified failure escaped the contract: ${error instanceof Error ? error.name : typeof error}`);
  }
  throw new Error("expected a refusal, but the call succeeded");
}

/** Refusals are bounded: no source path, traceback or Python exception text reaches the caller. */
function expectBoundedMessage(message: string): void {
  expect(message).not.toMatch(/Traceback|File "|\.py\b|\\\\/);
  expect(message.length).toBeLessThan(500);
}

/** Runs the runner script directly over stdin, with an explicit envelope, and reports its exit status and stderr. */
function runRaw(target: ConformanceTarget, envelope: object) {
  const result = spawnSync(target.python, [target.script], { input: JSON.stringify(envelope), encoding: "utf8", timeout: 60_000 });
  return { status: result.status, stderr: result.stderr ?? "", stdout: result.stdout ?? "" };
}
function stderrCode(stderr: string): string | undefined {
  try { return (JSON.parse(stderr.trim().split("\n").pop() ?? "") as { code?: string }).code; } catch { return undefined; }
}

export function describeContract(target: ConformanceTarget): void {
  describe.runIf(Boolean(target.python))(`${target.id} satisfies Meticule ML Backend Contract v1`, () => {
    const backend = target.create();
    const capabilities = backend.capabilities;
    const iterative = capabilities.iterative;

    // Real training is shared across checks, so each runner process is started a bounded number of times.
    let prepared: PreparedData; let model: TrainedModel; let predictions: unknown[];
    beforeAll(async () => {
      const snap = snapshotWith(target.id, target.classification, "classification");
      prepared = await backend.prepare(snap, ROWS, signal());
      model = await backend.train(snap, ROWS, prepared, signal());
      predictions = await backend.predict(snap, model.bytes, ROWS.slice(0, 12), model.environment, signal());
    }, FRAMEWORK_TRAINING_BUDGET);

    if (target.requiredInCi) {
      it("CI provides this runtime, so the suite cannot pass by skipping", () => {
        expect(target.python, `${target.id} interpreter must be configured in CI`).toBeTruthy();
      });
    }

    describe("identity and declared capabilities", () => {
      it("declares its id, exactly one artifact format, and a CPU fp32 path", () => {
        expect(backend.id).toBe(target.id);
        expect(capabilities.artifactFormats).toHaveLength(1);
        expect(capabilities.devices).toContain("cpu");
        expect(capabilities.precisions).toContain("fp32");
        expect(new Set(capabilities.preprocessing).size).toBe(capabilities.preprocessing.length);
      });
      it("is truthful about iteration: chunk methods exist exactly when iteration is declared", () => {
        const chunked = typeof backend.trainChunk === "function" && typeof backend.exportModel === "function";
        expect(chunked).toBe(iterative);
      });
      it("is truthful about models: every registered model fits inside the declared capabilities", () => {
        const models = listModels(target.id);
        expect(models.length).toBeGreaterThan(0);
        for (const m of models) {
          expect(capabilities.tasks).toContain(m.taskType);
          for (const type of m.inputs.types) expect(capabilities.inputs).toContain(type);
          for (const device of m.capabilities.devices ?? []) expect(capabilities.devices).toContain(device);
          if (m.capabilities.mixedPrecision) expect(capabilities.precisions).toContain("mixed");
          if (m.capabilities.iterative) expect(iterative).toBe(true);
        }
      });
      it("is truthful about tasks and inputs: each declared one is used by a registered model", () => {
        const models = listModels(target.id);
        expect([...capabilities.tasks].sort()).toEqual([...new Set(models.map(m => m.taskType))].sort());
        expect([...capabilities.inputs].sort()).toEqual([...new Set(models.flatMap(m => m.inputs.types))].sort());
      });
      it("registers its configured classification and regression models", () => {
        expect(getModel(target.id, target.classification.algorithm).taskType).toBe("classification");
        expect(getModel(target.id, target.regression.algorithm).taskType).toBe("regression");
      });
    });

    describe("validation refuses what the backend cannot honour, before any runner starts", () => {
      it("refuses an algorithm it does not register", () => {
        expect(() => backend.validate(snapshotWith(target.id, { algorithm: "arbitrary_import", hyperparameters: {} }, "classification"))).toThrow(ConflictError);
      });
      it("refuses a model registered under another backend", () => {
        const foreign = listModels().find(m => m.backend !== target.id);
        expect(foreign, "another backend must be registered for this check").toBeDefined();
        expect(() => backend.validate(snapshotWith(target.id, { algorithm: foreign!.id, hyperparameters: {} }, "classification"))).toThrow(ConflictError);
      });
      it("refuses preprocessing the runner does not interpret", () => {
        const snap = withSteps(snapshotWith(target.id, target.classification, "classification"), [{ operation: "bogus_transform" }]);
        expect(() => backend.validate(snap)).toThrow(ConflictError);
        expect(() => backend.validate(snap)).toThrow(/bogus_transform/);
      });
      it("accepts a numeric pipeline built from its declared operations", () => {
        const base = snapshotWith(target.id, target.classification, "classification");
        for (const op of capabilities.preprocessing.filter(o => ["impute_median", "impute_mean", "standard_scale", "min_max_scale"].includes(o))) {
          expect(() => backend.validate(withSteps(base, [{ operation: op }])), op).not.toThrow();
        }
      });
      it("refuses a device it cannot execute and accepts one it declares", () => {
        for (const device of ["cpu", "cuda"] as const) {
          const accepted = capabilities.devices.includes(device);
          const snap = (() => { try { return withPolicy(snapshotWith(target.id, target.classification, "classification"), { device, precision: "fp32" }); } catch { return null; } })();
          if (!iterative) {
            // A one-shot backend has no training policy at all, so even its CPU path is refused when a policy is supplied.
            expect(() => backend.validate(withPolicy(snapshotWith(target.id, target.classification, "classification"), { device, precision: "fp32" }))).toThrow(ConflictError);
            continue;
          }
          if (accepted) { expect(() => backend.validate(snap!)).not.toThrow(); continue; }
          expect(() => backend.validate(snap!)).toThrow(/does not support device cuda/);
        }
      });
      it("refuses mixed precision unless it declares it", () => {
        const declared = capabilities.precisions.includes("mixed");
        const make = () => withPolicy(snapshotWith(target.id, target.classification, "classification"), { device: "cuda", precision: "mixed" });
        if (declared && capabilities.devices.includes("cuda")) { expect(() => backend.validate(make())).not.toThrow(); return; }
        expect(() => backend.validate(make())).toThrow(ConflictError);
      });
      it("refuses an early-stopping policy unless the backend is iterative", () => {
        const make = () => withPolicy(snapshotWith(target.id, target.classification, "classification"), { earlyStopping: { patience: 2, minDelta: 0 } });
        if (iterative) expect(() => backend.validate(make())).not.toThrow();
        else expect(() => backend.validate(make())).toThrow(ConflictError);
      });
    });

    describe("prepare, train, evaluate and predict lifecycle", () => {
      it("prepares disjoint, complete partitions", () => {
        const all = [...prepared.indices.train, ...prepared.indices.validation, ...prepared.indices.test];
        expect(new Set(all).size).toBe(ROWS.length);
        expect(all.length).toBe(ROWS.length);
        for (const p of [prepared.indices.train, prepared.indices.validation, prepared.indices.test]) expect(p.length).toBeGreaterThan(0);
      });
      it("produces an artifact that declares its format, runtime and provenance", () => {
        expect(model.format).toBe(capabilities.artifactFormats[0]);
        expect(model.environment.artifact_format).toBe(capabilities.artifactFormats[0]);
        expect(model.environment.backend).toBe(target.id);
        expect(model.environment.adapter_sha256).toMatch(/^[a-f0-9]{64}$/);
        expect(model.bytes.length).toBeGreaterThan(0);
      });
      it("predicts one output per row", () => {
        expect(predictions).toHaveLength(12);
      });
      it("evaluates every partition with finite, declared metrics", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const metrics = await backend.evaluate(snap, ROWS, prepared, model, signal());
        for (const partition of ["train", "validation", "test"]) expect(metrics.some(m => m.partition === partition)).toBe(true);
        for (const m of metrics) expect(Number.isFinite(m.value)).toBe(true);
      });
      it("is deterministic for the same seed and split", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const again = await backend.train(snap, ROWS, prepared, signal());
        const second = await backend.predict(snap, again.bytes, ROWS.slice(0, 12), again.environment, signal());
        expect(second).toEqual(predictions);
      }, FRAMEWORK_TRAINING_BUDGET);
      it("trains and predicts a regression model", async () => {
        const snap = snapshotWith(target.id, target.regression, "regression");
        const p = await backend.prepare(snap, ROWS, signal());
        const m = await backend.train(snap, ROWS, p, signal());
        const out = await backend.predict(snap, m.bytes, ROWS.slice(0, 5), m.environment, signal());
        expect(out).toHaveLength(5);
        for (const v of out) expect(Number.isFinite(v as number)).toBe(true);
      }, FRAMEWORK_TRAINING_BUDGET);
      it("runs a contract v2 chronological split end to end: prepare, train, evaluate and predict", async () => {
        const snap = chronologicalSnapshot(target.id, target.classification, "classification");
        expect(contractFor(snap, "classification")).toBe(ML_BACKEND_CONTRACT_V2);
        const fresh = target.create(), calls = vi.spyOn(fresh as unknown as { call(...args: unknown[]): Promise<unknown> }, "call");
        const p = await fresh.prepare(snap, CHRONO_ROWS, signal());
        expectTimeOrdered(p);
        const m = await fresh.train(snap, CHRONO_ROWS, p, signal());
        const metrics = await fresh.evaluate(snap, CHRONO_ROWS, p, m, signal());
        for (const partition of ["train", "validation", "test"]) expect(metrics.some(x => x.partition === partition)).toBe(true);
        const out = await fresh.predict(snap, m.bytes, CHRONO_ROWS.slice(0, 7), m.environment, signal());
        expect(out).toHaveLength(7);
        for (const v of out) expect(["a", "b"]).toContain(v);
        // Every runner call (training may be chunked) spoke v2 and the runner echoed it.
        expect(calls.mock.calls.length).toBeGreaterThanOrEqual(4);
        for (const r of calls.mock.results) expect(((await r.value) as { contract: string }).contract).toBe(ML_BACKEND_CONTRACT_V2);
      }, FRAMEWORK_TRAINING_BUDGET);
    });

    describe("artifacts: compatibility is enforced, provenance is not", () => {
      it("refuses an artifact recorded under a different runtime", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const bad = { ...model.environment, numpy: "0.0.0" };
        const result = await refusal(backend.predict(snap, model.bytes, ROWS.slice(0, 3), bad, signal()));
        expect(result.category).toBe("incompatible_artifact");
        expectBoundedMessage(result.message);
      });
      it("refuses corrupt artifact bytes as an incompatible artifact", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const result = await refusal(backend.predict(snap, randomBytes(256), ROWS.slice(0, 3), model.environment, signal()));
        expect(result.category).toBe("incompatible_artifact");
        expectBoundedMessage(result.message);
      });
      it("does not treat provenance differences as incompatibility", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const sameRuntime = { ...model.environment, platform: "another-platform", adapter_sha256: "0".repeat(64) };
        const out = await backend.predict(snap, model.bytes, ROWS.slice(0, 3), sameRuntime, signal());
        expect(out).toHaveLength(3);
      });
    });

    describe("bounded failures and cancellation", () => {
      it("reports a cancelled call as cancelled and never starts a runner", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const aborted = new AbortController(); aborted.abort();
        const result = await refusal(backend.prepare(snap, ROWS, aborted.signal));
        expect(result.category).toBe("cancelled");
      });
      it("stops a runner that exceeds its time budget and reports it as timed out", async () => {
        const snap = snapshotWith(target.id, target.classification, "classification");
        const result = await refusal(target.create(1).prepare(snap, ROWS, signal()));
        expect(result.category).toBe("timed_out");
        expectBoundedMessage(result.message);
      }, 60_000);
      it("refuses a runner request from another contract before doing any work", () => {
        const result = runRaw(target, { command: "prepare", contract: "meticule-ml-backend/v0" });
        expect(result.status).not.toBe(0);
        expect(stderrCode(result.stderr)).toBe("unsupported_contract");
      });
      it("accepts the current contract and never reports it as a contract refusal", () => {
        const result = runRaw(target, { command: "unknown_command", contract: ML_BACKEND_CONTRACT });
        expect(stderrCode(result.stderr)).not.toBe("unsupported_contract");
      });
      it("enforces v1 semantics on prepare and predict and v2 task agreement before computation", () => {
        const snapshot = snapshotWith(target.id, target.classification, "classification");
        // A timestamp column is Phase 4 vocabulary, so a v1 envelope carrying it is refused before any work (A5).
        const extended = { ...snapshot, dataset: { ...snapshot.dataset, schema: { ...snapshot.dataset.schema, columns: [...snapshot.dataset.schema.columns, { name: "at", type: "timestamp", nullable: false }] } } };
        for (const command of ["prepare", "predict"]) {
          const result = runRaw(target, { command, contract: ML_BACKEND_CONTRACT, snapshot: extended, rows: ROWS });
          expect(result.status).toBe(1); expect(stderrCode(result.stderr)).toBe("unsupported_contract");
        }
        // v2 names the algorithm's task: another task (even one the runner implements) or an unknown algorithm is refused.
        for (const change of [{ task: "regression", snapshot }, { task: "embedding", snapshot }, { task: "classification", snapshot: { ...snapshot, algorithm: "unknown" } }]) {
          const result = runRaw(target, { command: "prepare", contract: ML_BACKEND_CONTRACT_V2, rows: ROWS, ...change });
          expect(result.status).toBe(1); expect(stderrCode(result.stderr)).toBe("unsupported_task");
        }
        const accepted = runRaw(target, { command: "prepare", contract: ML_BACKEND_CONTRACT_V2, task: "classification", snapshot, rows: ROWS });
        expect(accepted.status).toBe(0); expect(JSON.parse(accepted.stdout).contract).toBe(ML_BACKEND_CONTRACT_V2);
      }, 180_000);
    });

    describe("declared preprocessing is executed with the platform's semantics", () => {
      it("has a conformance expectation for every operation in the platform registry", () => {
        expect(Object.keys(OPERATION_EXPECTATIONS).sort()).toEqual(Object.keys(OPERATIONS).sort());
      });
      const declared = (role: string) => capabilities.preprocessing.find(o => OPERATION_EXPECTATIONS[o]?.role === role);
      const step = (operation: string, categorical: string) => operation === "ordinal_encode"
        ? { operation, parameters: { categories: { [categorical]: categorical === "kind" ? ["a", "b", "c"] : ["p", "q"] } } }
        : { operation };
      /** Trains the declared regression model on one pipeline, and returns a predictor plus the training partition's rows. */
      async function trained(features: string[], steps: { operation: string; parameters?: Record<string, unknown> }[]) {
        const snap = snapshotFor({ backend: target.id, algorithm: target.regression.algorithm, columns: PREP_COLUMNS, target: "y", features, steps, hyperparameters: target.regression.hyperparameters, rowCount: PREP_ROWS.length }) as Snapshot;
        backend.validate(snap);
        const prepared = await backend.prepare(snap, PREP_ROWS, signal());
        const model = await backend.train(snap, PREP_ROWS, prepared, signal());
        const predict = async (rows: Row[]) => {
          const out = await backend.predict(snap, model.bytes, rows, model.environment, signal()) as number[];
          expect(out).toHaveLength(rows.length);
          for (const v of out) expect(Number.isFinite(v)).toBe(true);
          return out;
        };
        return { predict, training: prepared.indices.train.map(i => PREP_ROWS[i]!) };
      }

      for (const op of capabilities.preprocessing) {
        it(`executes ${op} and it behaves as the platform defines it`, async () => {
          const expectation = OPERATION_EXPECTATIONS[op];
          expect(expectation, `${op} is declared but has no conformance expectation`).toBeDefined();
          switch (expectation!.role) {
            case "numeric_imputer": {
              // A missing value must predict exactly as the training statistic the operation fills with.
              const { predict, training } = await trained(["x", "xn"], [step(op, "")]);
              const fill = expectation!.fill!(training.map(r => r.xn).filter(v => v !== null) as number[]) as number;
              const [missing, filled, shifted] = await predict([{ x: 1, xn: null }, { x: 1, xn: fill }, { x: 1, xn: fill + 5 }]);
              expect(close(missing!, filled!), `${op}: a missing value is not filled with the training statistic`).toBe(true);
              expect(close(filled!, shifted!), `${op}: the model ignores the imputed feature, so the check is vacuous`).toBe(false);
              break;
            }
            case "numeric_transform": {
              const { predict } = await trained(["x", "w"], [step(op, "")]);
              const [low, high] = await predict([{ x: 0.5, w: 1 }, { x: 8, w: 3 }]);
              expect(close(low!, high!), `${op}: predictions do not depend on the transformed features`).toBe(false);
              break;
            }
            case "categorical_encoder": {
              // Categories stay distinguishable, and a category never seen in training is handled, not fatal.
              const { predict } = await trained(["x", "kind"], [step(op, "kind")]);
              const [a, b] = await predict([{ x: 1, kind: "a" }, { x: 1, kind: "b" }, { x: 1, kind: "never-seen" }]);
              expect(close(a!, b!), `${op}: distinct categories encode identically`).toBe(false);
              break;
            }
            case "categorical_imputer": {
              const encoder = declared("categorical_encoder");
              expect(encoder, `${op} is declared without any declared encoder to follow it`).toBeDefined();
              const { predict, training } = await trained(["x", "kindn"], [step(op, "kindn"), step(encoder!, "kindn")]);
              const fill = expectation!.fill!(training.map(r => r.kindn).filter(v => v !== null) as string[]) as string;
              const other = fill === "p" ? "q" : "p";
              const [missing, filled, different] = await predict([{ x: 1, kindn: null }, { x: 1, kindn: fill }, { x: 1, kindn: other }]);
              expect(close(missing!, filled!), `${op}: a missing category is not filled with the most frequent training category`).toBe(true);
              expect(close(filled!, different!), `${op}: the model ignores the imputed feature, so the check is vacuous`).toBe(false);
              break;
            }
            case "feature_selector": {
              // A feature that is constant in training carries no information and must not influence predictions.
              const { predict } = await trained(["x", "c"], [step(op, "")]);
              const [base, changedConstant, changedSignal] = await predict([{ x: 1, c: 0 }, { x: 1, c: 999 }, { x: 5, c: 0 }]);
              expect(close(base!, changedConstant!), `${op}: a zero-variance training feature still influences predictions`).toBe(true);
              expect(close(base!, changedSignal!), `${op}: predictions do not depend on the retained feature`).toBe(false);
              break;
            }
          }
        }, FRAMEWORK_TRAINING_BUDGET);
      }
    });

    describe("declared scores are produced with the promised shape and meaning", () => {
      const scaler = capabilities.preprocessing.find(o => OPERATION_EXPECTATIONS[o]?.role === "numeric_transform");
      const scoreSnapshot = (algorithm: string, label: "bin" | "tri", metrics: string[]) => snapshotFor({
        backend: target.id, algorithm, columns: SCORE_COLUMNS, target: label, features: ["x", "w"], steps: scaler ? [{ operation: scaler }] : [],
        hyperparameters: algorithm === target.classification.algorithm ? target.classification.hyperparameters : {}, evaluation: { metrics }, stratify: true, rowCount: SCORE_ROWS.length,
      }) as Snapshot;

      it("declares no scores for any regression model", () => {
        for (const m of listModels(target.id).filter(model => model.taskType === "regression")) expect(m.capabilities.scores, m.id).toBe("none");
      });

      for (const m of listModels(target.id).filter(model => model.taskType === "classification")) {
        const scores = m.capabilities.scores;
        if (scores === "none") {
          it(`${m.id} declares no scores, so score metrics are refused before any runner starts`, () => {
            expect(() => backend.validate(scoreSnapshot(m.id, "bin", ["roc_auc"]))).toThrow(ConflictError);
            expect(() => backend.validate(scoreSnapshot(m.id, "bin", ["log_loss"]))).toThrow(ConflictError);
          });
          continue;
        }
        // Probabilities are promised for any number of classes; a decision function only for binary targets.
        const targets: ("bin" | "tri")[] = scores === "probability" ? ["bin", "tri"] : ["bin"];
        const metrics = scores === "probability" ? ["accuracy", "roc_auc", "log_loss"] : ["accuracy", "roc_auc"];
        for (const label of targets) {
          it(`${m.id} produces valid ${scores} scores for a ${label === "bin" ? "binary" : "three-class"} target`, async () => {
            const snap = scoreSnapshot(m.id, label, metrics);
            const prepared = await backend.prepare(snap, SCORE_ROWS, signal());
            const model = await backend.train(snap, SCORE_ROWS, prepared, signal());
            // roc_auc and log_loss are computed from the model's real scores. For three classes both fail unless the
            // probability matrix has one column per training class and each row sums to one, so passing verifies its shape.
            const evaluated = await backend.evaluate(snap, SCORE_ROWS, prepared, model, signal());
            for (const partition of ["train", "validation", "test"]) for (const name of metrics) {
              const metric = evaluated.find(e => e.partition === partition && e.name === name);
              expect(metric, `${m.id} ${partition} ${name}`).toBeDefined();
              expect(Number.isFinite(metric!.value)).toBe(true);
              if (name === "roc_auc") expect(metric!.value >= 0 && metric!.value <= 1, `${m.id} ${partition} roc_auc out of range`).toBe(true);
              if (name === "log_loss") expect(metric!.value, `${m.id} ${partition} log_loss`).toBeGreaterThan(0);
            }
            // Constant or meaningless scores give a training roc_auc of 0.5; real scores on a learnable target do better.
            expect(evaluated.find(e => e.partition === "train" && e.name === "roc_auc")!.value, `${m.id} scores are uninformative`).toBeGreaterThan(0.6);
            const classes = new Set(SCORE_ROWS.map(r => r[label]));
            const predicted = await backend.predict(snap, model.bytes, SCORE_ROWS.slice(0, 10), model.environment, signal());
            for (const p of predicted) expect(classes.has(p as string), `${m.id} predicted a label it was never trained on`).toBe(true);
          }, FRAMEWORK_TRAINING_BUDGET);
        }
        if (scores === "decision") {
          it(`${m.id} declares a decision function, not probabilities, so log_loss is refused`, () => {
            expect(() => backend.validate(scoreSnapshot(m.id, "bin", ["log_loss"]))).toThrow(ConflictError);
          });
        }
      }
    });

    if (iterative) {
      describe("checkpoints: durable, resumable and fenced to their configuration", () => {
        const chunkSnap = () => snapshotWith(target.id, { algorithm: target.classification.algorithm, hyperparameters: { ...target.classification.hyperparameters, learningRate: 0.01 } }, "classification");
        const chunk = backend as MlBackend & Required<Pick<MlBackend, "trainChunk" | "exportModel">>;

        it("creates a checkpoint per chunk and resumes it to the next epochs", async () => {
          const snap = chunkSnap();
          const p = await backend.prepare(snap, ROWS, signal());
          const first = await chunk.trainChunk(snap, ROWS, p, null, 2, signal());
          expect(first.completedEpoch).toBe(2);
          expect(first.checkpointFormat).toBe(capabilities.checkpointFormats[0]);
          expect(first.epochs.map(e => e.epoch)).toEqual([1, 2]);
          const second = await chunk.trainChunk(snap, ROWS, p, first.checkpoint, 2, signal());
          expect(second.completedEpoch).toBe(4);
          expect(second.epochs.map(e => e.epoch)).toEqual([3, 4]);
        }, FRAMEWORK_TRAINING_BUDGET);

        it("produces the same model whether training is chunked or uninterrupted", async () => {
          const snap = chunkSnap();
          const p = await backend.prepare(snap, ROWS, signal());
          const whole = await chunk.trainChunk(snap, ROWS, p, null, 4, signal());
          const a = await chunk.trainChunk(snap, ROWS, p, null, 2, signal());
          const b = await chunk.trainChunk(snap, ROWS, p, a.checkpoint, 2, signal());
          const exportedWhole = await chunk.exportModel(snap, whole.checkpoint, false, signal());
          const exportedChunked = await chunk.exportModel(snap, b.checkpoint, false, signal());
          const out1 = await backend.predict(snap, exportedWhole.bytes, ROWS.slice(0, 8), exportedWhole.environment, signal());
          const out2 = await backend.predict(snap, exportedChunked.bytes, ROWS.slice(0, 8), exportedChunked.environment, signal());
          expect(out2).toEqual(out1);
        }, FRAMEWORK_TRAINING_BUDGET);

        it("refuses a corrupt checkpoint as an invalid checkpoint", async () => {
          const snap = chunkSnap();
          const p = await backend.prepare(snap, ROWS, signal());
          const result = await refusal(chunk.trainChunk(snap, ROWS, p, randomBytes(512), 1, signal()));
          expect(result.category).toBe("invalid_checkpoint");
          expectBoundedMessage(result.message);
        }, FRAMEWORK_TRAINING_BUDGET);

        it("refuses a checkpoint written by a different training configuration", async () => {
          const writer = chunkSnap();
          const p = await backend.prepare(writer, ROWS, signal());
          const written = await chunk.trainChunk(writer, ROWS, p, null, 1, signal());
          const other = snapshotWith(target.id, { algorithm: target.classification.algorithm, hyperparameters: { ...target.classification.hyperparameters, learningRate: 0.02 } }, "classification");
          const result = await refusal(chunk.trainChunk(other, ROWS, p, written.checkpoint, 1, signal()));
          expect(result.category).toBe("incompatible_checkpoint");
          expectBoundedMessage(result.message);
        }, FRAMEWORK_TRAINING_BUDGET);
      });
    }

    const cudaDeclared = capabilities.devices.includes("cuda");
    if (cudaDeclared && iterative) {
      describe("CUDA is executed when present and refused explicitly when not", () => {
        it("either trains on CUDA with CUDA recorded as the actual device, or refuses as device_unavailable", async () => {
          const snap = withPolicy(snapshotWith(target.id, target.classification, "classification"), { device: "cuda", precision: "fp32" });
          const p = await backend.prepare(snap, ROWS, signal());
          try {
            const trained = await backend.train(snap, ROWS, p, signal());
            expect(trained.environment.device_actual).toBe("cuda");
          } catch (error) {
            expect(error).toBeInstanceOf(MlBackendFailure);
            expect((error as MlBackendFailure).category).toBe("device_unavailable");
          }
        }, FRAMEWORK_TRAINING_BUDGET);
      });
    }
  });
}
