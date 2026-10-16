import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2, contractFor } from "../../src/ml/contract.js";
import { SplitDefinition, partitionSizes, type ColumnType } from "../../src/ml/domain.js";
import { TASKS, TASK_TYPES, maxPredictionRows, type TaskType } from "../../src/ml/tasks.js";
import { snapshotFor } from "./ml-fixtures.js";

type TableRow = { id: string; task: TaskType; split: { strategy: string; timeColumn?: string }; schema: { columns: { name: string; type: ColumnType }[]; target?: string }; contract: string };
// The shared A5 table: ml/ml_runtime_common.py uses_extensions is checked against it by check_contract.py.
const table = (JSON.parse(readFileSync(resolve("tests/fixtures/contract-table.json"), "utf8")) as { rows: TableRow[] }).rows;
const snapshotOf = (row: TableRow) => snapshotFor({
  columns: row.schema.columns, target: row.schema.target ?? null, features: row.schema.columns.map(c => c.name).filter(name => name !== row.schema.target),
  ...(row.split.strategy === "random" ? {} : { split: row.split }),
});

describe("task registry", () => {
  it("defines the M1 tasks and the rules each one imposes (spec sec. 4)", () => {
    expect(TASK_TYPES).toEqual(["classification", "regression", "forecasting", "anomaly_detection", "embedding"]);
    const rules = Object.fromEntries(TASK_TYPES.map(task => { const { output: _, ...rest } = TASKS[task]; return [task, rest]; }));
    const both = ["random", "chronological"];
    expect(rules).toEqual({
      classification: { target: "required", targetTypes: ["number", "string", "boolean"], splits: both, crossValidation: true, predictionIncludesTarget: false, outputs: "row", maxPredictionRows: 1000 },
      regression: { target: "required", targetTypes: ["number"], splits: both, crossValidation: true, predictionIncludesTarget: false, outputs: "row", maxPredictionRows: 1000 },
      forecasting: { target: "required", targetTypes: ["number"], splits: ["chronological"], crossValidation: false, predictionIncludesTarget: true, outputs: "horizon", maxPredictionRows: 5000 },
      anomaly_detection: { target: "optional", targetTypes: ["boolean"], splits: both, crossValidation: false, predictionIncludesTarget: false, outputs: "row", maxPredictionRows: 1000 },
      embedding: { target: "optional", targetTypes: ["number", "string", "boolean"], splits: both, crossValidation: false, predictionIncludesTarget: false, outputs: "row", maxPredictionRows: 256 },
    });
  });

  it("validates one prediction with the task's output schema", () => {
    const step = { step: 1, timestamp: "2024-03-01T00:00:00+01:00", value: 2.5 };
    const accepted: Record<TaskType, unknown[]> = {
      classification: ["spam", 3, false, "x".repeat(10_000)], regression: [0, -1.5, 1e300],
      forecasting: [step, { ...step, step: 100 }], anomaly_detection: [{ score: -0.2, anomaly: false }, { score: 3, anomaly: true }],
      embedding: [[0.25], [0, -1, 2], new Array(4096).fill(0.5)],
    };
    const refused: Record<TaskType, unknown[]> = {
      classification: [null, {}, [], Number.NaN, Number.POSITIVE_INFINITY, "x".repeat(10_001)],
      regression: ["1", true, null, Number.NaN, Number.NEGATIVE_INFINITY],
      forecasting: [{ ...step, step: 0 }, { ...step, step: 1.5 }, { ...step, timestamp: "2024-03-01T00:00:00" }, { ...step, timestamp: "2024-02-30T00:00:00Z" }, { ...step, value: Number.NaN }, { ...step, extra: 1 }, { step: 1, value: 1 }],
      anomaly_detection: [{ score: 1 }, { score: Number.NaN, anomaly: true }, { score: 1, anomaly: 1 }, { score: 1, anomaly: true, label: "x" }, 0.5],
      embedding: [[], new Array(4097).fill(0), [Number.NaN], ["1"], 0.5],
    };
    for (const task of TASK_TYPES) {
      for (const value of accepted[task]) expect(TASKS[task].output.safeParse(value).success, `${task} accepts ${JSON.stringify(value)?.slice(0, 40)}`).toBe(true);
      for (const value of refused[task]) expect(TASKS[task].output.safeParse(value).success, `${task} refuses ${JSON.stringify(value)?.slice(0, 40)}`).toBe(false);
    }
  });

  it("caps prediction rows per task, lowered for media inputs (C5)", () => {
    expect(maxPredictionRows("classification", ["number", "string"])).toBe(1000);
    expect(maxPredictionRows("classification", ["text"])).toBe(1000);
    expect(maxPredictionRows("embedding", ["number"])).toBe(256);
    expect(maxPredictionRows("embedding", ["text"])).toBe(64);
    expect(maxPredictionRows("classification", ["image", "number"])).toBe(32);
    expect(maxPredictionRows("embedding", ["text", "image"])).toBe(32);
    expect(maxPredictionRows("regression", ["image", "audio"])).toBe(16);
    expect(maxPredictionRows("forecasting", ["timestamp", "number"])).toBe(5000);
  });
});

describe("contract selection", () => {
  it("chooses v1 only for Phase 1-3 semantics, for every row of the shared table (A5)", () => {
    expect(table.length).toBeGreaterThanOrEqual(10);
    expect(new Set(table.map(row => row.contract))).toEqual(new Set([ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2]));
    for (const row of table) expect(contractFor(snapshotOf(row), row.task), row.id).toBe(row.contract);
  });

  it("keeps a stored Phase 3 snapshot on v1, stratified and with a boolean regression target (A4)", () => {
    const stored = snapshotFor({ columns: [{ name: "x", type: "number" }, { name: "flag", type: "boolean" }], target: "flag", stratify: true });
    expect(contractFor(stored, "regression")).toBe(ML_BACKEND_CONTRACT);
    expect(contractFor(stored, "embedding")).toBe(ML_BACKEND_CONTRACT_V2);
  });
});

const python = process.env.ML_TEST_PYTHON;
// The runtime's own decisions on exactly what the TypeScript side serialises: one wrong mirror on either side fails here.
describe.runIf(Boolean(python))("contract and partitions across languages", () => {
  it("Python decides every snapshot, raw envelope and split size exactly as TypeScript does", () => {
    const snapshots = table.map(row => ({ id: row.id, task: row.task, snapshot: JSON.parse(JSON.stringify(snapshotOf(row))) as Record<string, unknown> }));
    const other = (task: TaskType) => TASK_TYPES.find(t => t !== task)!;
    // Raw envelopes as runner-backend sends them, and the refusals serve makes before any work (A1).
    const envelopes = table.flatMap((row, i) => {
      const { snapshot } = snapshots[i]!, v2 = { contract: ML_BACKEND_CONTRACT_V2, task: row.task, snapshot };
      const v1Outcome = contractFor(snapshotOf(row), row.task) === ML_BACKEND_CONTRACT ? null : "unsupported_contract";
      return [
        { id: `${row.id}/v1`, runnerTask: row.task, request: { contract: ML_BACKEND_CONTRACT, snapshot }, expected: v1Outcome },
        { id: `${row.id}/v1-unknown-algorithm`, runnerTask: null, request: { contract: ML_BACKEND_CONTRACT, snapshot }, expected: "unsupported_contract" },
        { id: `${row.id}/v2`, runnerTask: row.task, request: v2, expected: null },
        { id: `${row.id}/v2-other-task`, runnerTask: other(row.task), request: v2, expected: "unsupported_task" },
        { id: `${row.id}/v2-unknown-algorithm`, runnerTask: null, request: v2, expected: "unsupported_task" },
        { id: `${row.id}/v2-without-task`, runnerTask: row.task, request: { contract: ML_BACKEND_CONTRACT_V2, snapshot }, expected: "unsupported_task" },
      ];
    });
    const fractions = [[.6, .2, .2], [.7, .15, .15], [.8, .1, .1], [.34, .33, .33], [.1, .3, .6], [.9, .05, .05]];
    const splits = ["random", "chronological"].flatMap(strategy => fractions.flatMap(([train, validation, test]) => [1, 2, 3, 5, 7, 10, 19, 99, 100, 101, 997, 5001].map(n => {
      const split = SplitDefinition.parse({ id: "11111111-1111-4111-8111-111111111111", strategy, train, validation, test, ...(strategy === "chronological" ? { timeColumn: "at" } : {}) });
      return { id: `${strategy}/${train}/${validation}/${test}/${n}`, split, n };
    })));
    const out = JSON.parse(execFileSync(python!, ["-I", "-B", resolve("tests/python/contract_table.py")], {
      input: JSON.stringify({ snapshots, envelopes: envelopes.map(({ expected: _, ...e }) => e), splits }), encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1" },
    }).trim().split("\n").pop()!) as { usesExtensions: Record<string, boolean>; envelopes: Record<string, string | null>; sizes: Record<string, number[] | null> };

    expect(out.usesExtensions).toEqual(Object.fromEntries(table.map(row => [row.id, contractFor(snapshotOf(row), row.task) === ML_BACKEND_CONTRACT_V2])));
    expect(out.envelopes).toEqual(Object.fromEntries(envelopes.map(e => [e.id, e.expected])));
    expect(out.sizes).toEqual(Object.fromEntries(splits.map(({ id, split, n }) => {
      const sizes = partitionSizes(split, n);
      return [id, Math.min(sizes.train, sizes.validation, sizes.test) < 1 ? null : [sizes.train, sizes.validation, sizes.test]];
    })));
  }, 120_000);
});
