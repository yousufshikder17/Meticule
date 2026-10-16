import { beforeEach, describe, expect, it, vi } from "vitest";
import { RunnerBackend } from "../../src/ml/runner-backend.js";
import { callRunner } from "../../src/ml/runner-process.js";
import { defineModel, registerModels } from "../../src/ml/model-catalog.js";
import { COLUMN_TYPES, type Snapshot } from "../../src/ml/domain.js";
import { ML_BACKEND_CONTRACT, ML_BACKEND_CONTRACT_V2 } from "../../src/ml/contract.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import type { BackendCapabilities } from "../../src/ml/backend.js";
import { TASK_TYPES, embeddingDimension } from "../../src/ml/tasks.js";
import { snapshotFor } from "./ml-fixtures.js";

vi.mock("../../src/ml/runner-process.js", () => ({ callRunner: vi.fn() }));
const capabilities = { artifactFormats: ["fake"], checkpointFormats: [], devices: ["cpu"], precisions: ["fp32"], preprocessing: ["standard_scale"], iterative: false, tasks: TASK_TYPES, inputs: COLUMN_TYPES } as const;
registerModels([
  defineModel({ backend: "v2-test", id: "embed", taskType: "embedding", displayName: "Embed", parameters: {}, capabilities: { scores: "none", scaleSensitive: false, seeded: true }, inputs: { types: ["text"], min: 1, max: 1 }, learning: "pretrained" }),
  defineModel({ backend: "v2-test", id: "forecast", taskType: "forecasting", displayName: "Forecast", parameters: { horizon: { kind: "int", min: 1, max: 100, default: 2 }, seasonal_period: { kind: "int", min: 1, max: 100, default: 1 } }, capabilities: { scores: "none", scaleSensitive: false, seeded: true }, inputs: { types: ["timestamp"], min: 1, max: 1 } }),
  defineModel({ backend: "v2-test", id: "anomaly", taskType: "anomaly_detection", displayName: "Anomaly", parameters: {}, capabilities: { scores: "decision", scaleSensitive: false, seeded: true }, learning: "unsupervised" }),
]);
class FakeRunner extends RunnerBackend {
  constructor(caps: BackendCapabilities = capabilities) { super({ id: "v2-test", capabilities: caps, python: "unused", script: "unused", timeoutMs: 1000 }); }
  protected override requestExtras() { return { modelRoot: "test-root" }; }
}
const embed = (extra: Parameters<typeof snapshotFor>[0] = {}) => snapshotFor({ backend: "v2-test", algorithm: "embed", columns: [{ name: "x", type: "text" }], target: null, ...extra });
const forecast = (extra: Parameters<typeof snapshotFor>[0] = {}) => snapshotFor({ backend: "v2-test", algorithm: "forecast", columns: [{ name: "x", type: "timestamp" }, { name: "y", type: "number" }], split: { strategy: "chronological", timeColumn: "x" }, ...extra });
const signal = () => new AbortController().signal;
beforeEach(() => vi.mocked(callRunner).mockReset());

describe("v2 backend boundary", () => {
  it("checks optional targets, CV, metrics, preprocessing and embedding partition bounds before computation", async () => {
    const backend = new FakeRunner();
    expect(() => backend.validate(embed())).not.toThrow();
    expect(() => backend.validate(embed({ rowCount: 8334 }))).not.toThrow(); // floor(.6 * 8334) = 5000
    for (const [snapshot, message] of [
      [embed({ rowCount: 8335 }), /5000/], [embed({ stratify: true }), /Stratification/],
      [embed({ evaluation: { cv: { strategy: "kfold", folds: 2 } } }), /Cross-validation/],
      [embed({ evaluation: { metrics: ["retrieval_mrr"] } }), /needs a target/],
      [embed({ steps: [{ operation: "standard_scale" }] }), /tabular model/],
      [embed({ columns: [{ name: "x", type: "text" }, { name: "z", type: "text" }], features: ["x", "z"] }), /Feature count/],
    ] as [Snapshot, RegExp][]) await expect(backend.prepare(snapshot, [], signal())).rejects.toThrow(message);
    expect(callRunner).not.toHaveBeenCalled();
  });
  it("checks truthful task and input capabilities independently of model declarations", () => {
    expect(() => new FakeRunner({ ...capabilities, tasks: ["regression"] }).validate(embed())).toThrow(/support this task/);
    expect(() => new FakeRunner({ ...capabilities, inputs: ["number"] }).validate(embed())).toThrow(/feature column types/);
    expect(() => new FakeRunner().validate(embed({ columns: [{ name: "x", type: "number" }] }))).toThrow(/feature column types/);
  });
  it("enforces chronological and forecasting rules, including non-nullable history columns", () => {
    const backend = new FakeRunner();
    expect(() => backend.validate(forecast())).not.toThrow();
    for (const [snapshot, message] of [
      [forecast({ split: { strategy: "random" } }), /Split strategy/],
      [forecast({ split: { strategy: "chronological", timeColumn: "missing" } }), /timestamp column/],
      [forecast({ columns: [{ name: "x", type: "timestamp", nullable: true }, { name: "y", type: "number" }] }), /non-nullable timestamp/],
      [forecast({ columns: [{ name: "x", type: "timestamp" }, { name: "y", type: "number", nullable: true }] }), /non-nullable target/],
      [forecast({ target: null }), /requires a target/], [forecast({ rowCount: 3 }), /at least one row/],
      [forecast({ evaluation: { cv: { strategy: "kfold", folds: 2 } } }), /chronological split/],
    ] as [Snapshot, RegExp][]) expect(() => backend.validate(snapshot)).toThrow(message);
  });
  it("selects v2 and supplies the catalog task and backend extras in every request", async () => {
    vi.mocked(callRunner).mockResolvedValue({ indices: { train: [0], validation: [1], test: [2] }, environment: {} });
    await new FakeRunner().prepare(embed(), [], signal());
    expect(callRunner).toHaveBeenCalledWith(expect.objectContaining({ contract: ML_BACKEND_CONTRACT_V2, request: expect.objectContaining({ command: "prepare", task: "embedding", modelRoot: "test-root" }) }));
  });
  it("keeps the Phase 3 envelope for a v1 snapshot: contract v1 and no task", async () => {
    vi.mocked(callRunner).mockResolvedValue({ indices: { train: [0], validation: [1], test: [2] }, environment: {} });
    await new SklearnBackend("unused").prepare(snapshotFor(), [], signal());
    const call = vi.mocked(callRunner).mock.calls[0]![0];
    expect(call.contract).toBe(ML_BACKEND_CONTRACT); expect(Object.keys(call.request).sort()).toEqual(["command", "rows", "snapshot"]);
  });
  it("validates embedding dimensions without coercion and rejects malformed or wrongly sized vectors", async () => {
    for (const raw of [undefined, "", "0", "01", "1.0", "1e2", "+1", " 1", "1\n", "4097", "10000", 2]) expect(() => embeddingDimension({ embedding_dimension: raw } as Record<string, string>)).toThrow(/Invalid embedding/);
    expect(embeddingDimension({ embedding_dimension: "4096" })).toBe(4096);
    const backend = new FakeRunner(), s = embed();
    const predict = () => backend.predict(s, Buffer.from("x"), [{ x: "text" }], { embedding_dimension: "2" }, signal());
    vi.mocked(callRunner).mockResolvedValue({ predictions: [[1, 2]] }); expect(await predict()).toEqual([[1, 2]]);
    for (const predictions of [null, [], [[1]], [[1, 2, 3]], [[NaN, 1]], [["1", 2]], [[1, 2], [3, 4]]]) {
      vi.mocked(callRunner).mockResolvedValue({ predictions }); await expect(predict()).rejects.toMatchObject({ category: "invalid_response" });
    }
  });
  it("uses the forecast default or explicit horizon and requires contiguous ordered steps", async () => {
    const backend = new FakeRunner(), item = { step: 1, timestamp: "2024-01-01T00:00Z", value: 1 };
    const predict = (s = forecast()) => backend.predict(s, Buffer.from("x"), [{ x: "2023-12-31T00:00Z", y: 0 }], {}, signal());
    const predictions = [item, { ...item, step: 2 }];
    vi.mocked(callRunner).mockResolvedValue({ predictions }); expect(await predict()).toEqual(predictions);
    vi.mocked(callRunner).mockResolvedValue({ predictions: [item] }); expect(await predict(forecast({ hyperparameters: { horizon: 1 } }))).toEqual([item]);
    for (const bad of [[item], [item, item], [...predictions].reverse(), [item, { ...item, step: 2, value: Infinity }], [item, { ...item, step: 2, extra: true }]]) {
      vi.mocked(callRunner).mockResolvedValue({ predictions: bad }); await expect(predict()).rejects.toMatchObject({ category: "invalid_response" });
    }
  });
  it("validates anomaly objects before returning a successful prediction", async () => {
    const backend = new FakeRunner(), s = snapshotFor({ backend: "v2-test", algorithm: "anomaly", target: null });
    const predict = () => backend.predict(s, Buffer.from("x"), [{ x: 1 }], {}, signal());
    vi.mocked(callRunner).mockResolvedValue({ predictions: [{ score: .5, anomaly: false }] }); expect(await predict()).toEqual([{ score: .5, anomaly: false }]);
    vi.mocked(callRunner).mockResolvedValue({ predictions: [{ score: .5, anomaly: "false" }] }); await expect(predict()).rejects.toMatchObject({ category: "invalid_response" });
  });
});
