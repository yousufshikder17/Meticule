import { spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";
import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";
import { MlBackendFailure } from "../../src/ml/contract.js";
import { MlGovernanceService, GovernanceRefusal } from "../../src/ml/governance.js";
import { MlLineageService } from "../../src/ml/lineage.js";
import { MlModelService } from "../../src/ml/model-service.js";
import { MlOnnxService, parity, PARITY_TOLERANCE } from "../../src/ml/onnx.js";
import { MlTrainingService } from "../../src/ml/persistence.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { TrainingProcessor } from "../../src/ml/training-processor.js";
import { TrainingSnapshot } from "../../src/ml/domain.js";
import { LifecycleWorker } from "../../src/worker/worker.js";
import { approver, manager, outsider, pool, trainedModel, truncate, viewer, withStore } from "./mlops-fixtures.js";

const sklearnPython = process.env.ML_TEST_PYTHON, mlopsPython = process.env.ML_TEST_MLOPS_PYTHON;
const signal = () => new AbortController().signal;
beforeEach(truncate);
afterAll(() => pool.end());

/** Trains a real scikit-learn model through the durable lifecycle and registers it. */
async function sklearnModel(store: PostgresArtifactStore, algorithm: string, target: "number" | "string" | "int", options: { features?: "numeric" | "mixed"; hyperparameters?: Record<string, unknown> } = {}) {
  const backend = new SklearnBackend(sklearnPython!), backends = new Map([[backend.id, backend]]);
  const training = new MlTrainingService(pool, store), models = new MlModelService(pool, store, backends);
  const rows = Array.from({ length: 80 }, (_, i) => {
    const a = Math.sin(i) * 3, b = i % 7 === 0 ? null : Math.cos(i * 1.3) * 2, score = 2 * a - (b ?? 0) + .5;
    return { a, b, ...(options.features === "mixed" ? { c: i % 2 ? "x" : "y" } : {}), y: target === "number" ? score : target === "string" ? (score > 0 ? "high" : "low") : (score > 0 ? 1 : 0) };
  });
  const columns = [{ name: "a", type: "number" }, { name: "b", type: "number", nullable: true }, ...(options.features === "mixed" ? [{ name: "c", type: "string" }] : []), { name: "y", type: target === "number" || target === "int" ? "number" : "string" }];
  const dataset = await training.ingest(manager, { name: "onnx", schema: { id: randomUUID(), columns, target: "y" }, rows });
  const steps = [{ operation: "impute_median" }, { operation: "standard_scale" }, ...(options.features === "mixed" ? [{ operation: "impute_most_frequent" }, { operation: "one_hot_encode" }] : [])];
  const pipeline = await training.pipeline(manager, { name: "p", definition: { features: options.features === "mixed" ? ["a", "b", "c"] : ["a", "b"], steps } });
  const experiment = await training.experiment(manager, "onnx");
  const job = await training.queue(manager, { experimentId: experiment.id, spec: { datasetVersionId: dataset.id, featurePipelineId: pipeline.id, backend: "sklearn", algorithm, hyperparameters: options.hyperparameters ?? {}, seed: 3, split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: target !== "number" } } });
  await new LifecycleWorker(pool, { workerId: "onnx-test", leaseSeconds: 60, kind: "ml" }, new TrainingProcessor(pool, store, backends)).tick();
  const attempt = (await training.getJob(manager, job.id)).attempts[0];
  expect(attempt.status).toBe("COMPLETED");
  const registry = await models.registry(manager, { name: `onnx-${algorithm}-${target}-${randomUUID().slice(0, 4)}` });
  const version = await models.register(manager, registry.id, { trainingRunId: attempt.id });
  return { backend, models, version, registry, attempt, rows, dataset };
}

/** Builds a hostile ONNX model with the onnx package (never with the exporter under test). */
function hostile(kind: string): Buffer {
  const script = `
import base64, onnx, sys
from onnx import helper, TensorProto, numpy_helper
import numpy as np
kind = sys.argv[1]
x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])
y = helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])
opset = [helper.make_opsetid("", 17)]
if kind == "operator":
    nodes = [helper.make_node("Einsum", ["input", "w"], ["variable"], equation="ij,jk->ik")]
    graph = helper.make_graph(nodes, "g", [x], [y], [numpy_helper.from_array(np.ones((2, 1), dtype=np.float32), "w")])
elif kind == "domain":
    nodes = [helper.make_node("Evil", ["input"], ["variable"], domain="com.attacker")]
    graph = helper.make_graph(nodes, "g", [x], [y]); opset.append(helper.make_opsetid("com.attacker", 1))
elif kind == "opset":
    nodes = [helper.make_node("Identity", ["input"], ["variable"])]
    graph = helper.make_graph(nodes, "g", [x], [y]); opset = [helper.make_opsetid("", 21)]
elif kind == "external":
    w = numpy_helper.from_array(np.ones((2, 1), dtype=np.float32), "w")
    w.ClearField("raw_data"); w.data_location = TensorProto.EXTERNAL
    w.external_data.add(key="location", value="../../secrets.bin")
    nodes = [helper.make_node("MatMul", ["input", "w"], ["variable"])]
    graph = helper.make_graph(nodes, "g", [x], [y], [w])
model = helper.make_model(graph, opset_imports=opset)
sys.stdout.write(base64.b64encode(model.SerializeToString()).decode())
`;
  const r = spawnSync(mlopsPython!, ["-c", script, kind], { encoding: "utf8" });
  if (r.status !== 0) throw new Error(r.stderr);
  return Buffer.from(r.stdout, "base64");
}

describe("ONNX parity policy", () => {
  it("bounds regression error and requires every class label to agree", () => {
    expect(parity("regression", { predictions: [1, 100], probabilities: null }, { predictions: [1.00005, 100.09], probabilities: null })).toMatchObject({ passed: true });
    expect(parity("regression", { predictions: [1, 100], probabilities: null }, { predictions: [1.01, 100], probabilities: null })).toMatchObject({ passed: false });
    expect(parity("regression", { predictions: [1, 2], probabilities: null }, { predictions: [1], probabilities: null }).passed).toBe(false);
    expect(parity("classification", { predictions: ["a", "b"], probabilities: [[.9, .1], [.2, .8]] }, { predictions: ["a", "b"], probabilities: [[.90001, .09999], [.2, .8]] })).toMatchObject({ passed: true, labelAgreement: 1 });
    expect(parity("classification", { predictions: ["a", "b"], probabilities: null }, { predictions: ["a", "a"], probabilities: null })).toMatchObject({ passed: false, labelAgreement: .5 });
    expect(parity("classification", { predictions: [1], probabilities: [[.5, .5]] }, { predictions: [1], probabilities: [[.6, .4]] }).passed).toBe(false);
    expect(PARITY_TOLERANCE.regression).toEqual({ absolute: 1e-4, relative: 1e-3 });
  });
});

describe.runIf(Boolean(sklearnPython && mlopsPython))("ONNX export and ONNX Runtime inference", () => {
  const config = () => ({ python: mlopsPython!, timeoutMs: 120_000 });

  it("exports scikit-learn regression and classification pipelines that match native predictions", () => withStore(async store => {
    const onnx = new MlOnnxService(pool, store, config());
    for (const [algorithm, target, operator] of [["ridge", "number", "ai.onnx.ml:LinearRegressor"], ["random_forest_regressor", "number", "ai.onnx.ml:TreeEnsembleRegressor"], ["logistic_regression", "string", "ai.onnx.ml:LinearClassifier"], ["gradient_boosting_classifier", "int", "ai.onnx.ml:TreeEnsembleClassifier"], ["k_neighbors_classifier", "int", "Scan"]] as const) {
      const m = await sklearnModel(store, algorithm, target);
      const e = await onnx.export(manager, m.version.id, signal());
      expect(e.status, algorithm).toBe("VERIFIED");
      expect(e.parity).toMatchObject({ passed: true, rows: 16, task: target === "number" ? "regression" : "classification" });
      expect(e.operators).toContain(operator);
      expect(e.opsets[""]).toBeLessThanOrEqual(17); expect(e.signature.inputs[0]).toMatchObject({ name: "input", elemType: "float32", shape: [null, 2] });
      expect(e.producer).toMatchObject({ converter: "skl2onnx", skl2onnx: "1.20.0", onnx: "1.23.0", sklearn: "1.7.2" });
      if (target !== "number") expect(e.parity.labelAgreement).toBe(1);
      // ONNX Runtime and the native backend agree on new rows too (rows with a null go through the ONNX Imputer).
      const snapshot = TrainingSnapshot.parse(m.attempt.snapshot), rows = [{ a: .3, b: null }, { a: -2.5, b: 1.1 }, { a: 2, b: -1 }];
      const viaOnnx = await onnx.predict(manager.tenantId, e.content, snapshot, rows, signal());
      const native = await m.backend.predict(snapshot, await store.get(manager.tenantId, (await pool.query("SELECT content FROM ml_artifacts WHERE id=$1", [m.version.artifact_id])).rows[0].content), rows, m.attempt.environment, signal());
      if (target === "number") viaOnnx.forEach((v, i) => expect(Number(v)).toBeCloseTo(Number(native[i]), 3)); else expect(viaOnnx).toEqual(native);
      const bytes = await onnx.content(manager, e.id);
      expect(bytes.length).toBe(e.content.size);
      expect((await new MlLineageService(pool, store).modelVersion(viewer, m.version.id)).downstream.onnxExports.map(x => x.id)).toEqual([e.id]);
    }
  }), 600_000);

  it("records unsupported combinations instead of converting them, and keeps tenants and roles apart", () => withStore(async store => {
    const onnx = new MlOnnxService(pool, store, config());
    const mixed = await sklearnModel(store, "logistic_regression", "string", { features: "mixed" });
    expect(await onnx.export(manager, mixed.version.id, signal())).toMatchObject({ status: "UNSUPPORTED", content: null, failure: { category: "unsupported_capability" } });
    const other = await trainedModel(store); // a non-scikit-learn backend
    const registry = await other.models.registry(manager, { name: "native-only" });
    const version = await other.models.register(manager, registry.id, { trainingRunId: other.attempt.id });
    expect((await onnx.export(manager, version.id, signal())).status).toBe("UNSUPPORTED");
    await other.models.transition(manager, version.id, "REJECTED");
    await expect(onnx.export(manager, version.id, signal())).rejects.toThrow(/REGISTERED or READY/);
    await expect(onnx.export(outsider, mixed.version.id, signal())).rejects.toThrow(/not found/);
    await expect(onnx.export(viewer, mixed.version.id, signal())).rejects.toThrow(/role/);
    expect(await onnx.list(viewer, mixed.version.id)).toHaveLength(1);
    await expect(pool.query("UPDATE ml_onnx_exports SET status='VERIFIED' WHERE model_version_id=$1", [mixed.version.id])).rejects.toThrow();
    // Public storage: an export the 16 MB blob limit refuses is recorded as unsupported, keeping the limit's category.
    class LimitedStore extends PostgresArtifactStore {
      override async put(tenantId: string, bytes: Uint8Array, mediaType: string) {
        if (mediaType === "application/onnx") throw new MlBackendFailure("output_limit_exceeded", "ML artifact exceeds the 16 MB public storage limit");
        return super.put(tenantId, bytes, mediaType);
      }
    }
    const ridge = await sklearnModel(store, "ridge", "number");
    expect(await new MlOnnxService(pool, new LimitedStore(pool), config()).export(manager, ridge.version.id, signal())).toMatchObject({ status: "UNSUPPORTED", content: null, failure: { category: "output_limit_exceeded", message: expect.stringMatching(/16 MB/) } });
  }), 300_000);

  it("refuses hostile ONNX models at the runtime gate before a session is created", () => withStore(async store => {
    const m = await sklearnModel(store, "ridge", "number");
    const snapshot = TrainingSnapshot.parse(m.attempt.snapshot), onnx = new MlOnnxService(pool, store, config());
    for (const kind of ["operator", "domain", "opset", "external"]) {
      const content = await store.put(manager.tenantId, hostile(kind), "application/onnx");
      await expect(onnx.predict(manager.tenantId, content, snapshot, [{ a: 1, b: 2 }], signal()), kind).rejects.toMatchObject({ category: "incompatible_artifact", runnerCode: "onnx_model_refused" });
    }
    const garbage = await store.put(manager.tenantId, Buffer.from("not onnx"), "application/onnx");
    await expect(onnx.predict(manager.tenantId, garbage, snapshot, [{ a: 1, b: 2 }], signal())).rejects.toMatchObject({ category: "incompatible_artifact", runnerCode: "onnx_model_refused" });
  }), 300_000);

  it("lets a registry require a verified ONNX export before READY", () => withStore(async store => {
    const m = await sklearnModel(store, "ridge", "number");
    const governance = new MlGovernanceService(pool, store);
    await governance.setPolicy(manager, m.registry.id, { requireVerifiedOnnx: true });
    await expect(m.models.transition(approver, m.version.id, "READY")).rejects.toSatisfy((e: unknown) => e instanceof GovernanceRefusal && e.reasons[0]!.code === "onnx_export_required");
    await new MlOnnxService(pool, store, config()).export(manager, m.version.id, signal());
    expect((await m.models.transition(approver, m.version.id, "READY")).status).toBe("READY");
  }), 300_000);
});
