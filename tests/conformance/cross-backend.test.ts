import { randomBytes } from "node:crypto";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import type { MlBackend, TrainedModel } from "../../src/ml/backend.js";
import { MlBackendFailure } from "../../src/ml/contract.js";
import { JaxBackend } from "../../src/ml/jax-backend.js";
import { PytorchBackend } from "../../src/ml/pytorch-backend.js";
import { SklearnBackend } from "../../src/ml/sklearn-backend.js";
import { TensorflowBackend } from "../../src/ml/tensorflow-backend.js";
import { ROWS, signal, snapshotWith } from "./contract-suite.js";

// Cross-backend refusal: no framework may consume another framework's artifact or checkpoint. Each pair runs only when both
// interpreters are configured, and CI provides the pairs it can host (see .github/workflows/ci.yml).
const sklearnPython = process.env.ML_TEST_PYTHON ?? "";
const torchPython = process.env.ML_TEST_PYTORCH_PYTHON ?? "";
const tensorflowPython = process.env.ML_TEST_TENSORFLOW_PYTHON ?? "";
const jaxPython = process.env.ML_TEST_JAX_PYTHON ?? "";
const quick = { epochs: 3, hiddenLayers: [8], batchSize: 16, learningRate: 0.01 };
const CLASSIFICATION = { algorithm: "mlp_classifier", hyperparameters: quick };

async function refusalOf(promise: Promise<unknown>): Promise<string> {
  try { await promise; } catch (error) {
    if (error instanceof MlBackendFailure) return error.category;
    throw error;
  }
  throw new Error("expected a refusal, but the call succeeded");
}

/**
 * The snapshot a receiving backend is asked under. The platform picks a backend from the stored snapshot, so a foreign
 * artifact can only reach another runner under a snapshot that names that runner. The runner must still refuse it.
 */
function receiverSnapshot(backend: MlBackend) {
  return backend.id === "sklearn"
    ? snapshotWith("sklearn", { algorithm: "logistic_regression", hyperparameters: {} }, "classification")
    : snapshotWith(backend.id, CLASSIFICATION, "classification");
}

/** Trains one attempt with a backend; its artifact and recorded environment are what another backend is handed. */
async function trainedArtifact(backend: MlBackend, algorithm: string, hyperparameters: Record<string, unknown>): Promise<{ model: TrainedModel; snap: ReturnType<typeof snapshotWith> }> {
  const snap = snapshotWith(backend.id, { algorithm, hyperparameters }, "classification");
  const prepared = await backend.prepare(snap, ROWS, signal());
  return { model: await backend.train(snap, ROWS, prepared, signal()), snap };
}

describe.runIf(Boolean(sklearnPython && torchPython))("scikit-learn and PyTorch refuse each other's artifacts", () => {
  it("refuses a scikit-learn artifact in PyTorch and a PyTorch artifact in scikit-learn", async () => {
    const sklearn = new SklearnBackend(sklearnPython);
    const torch = new PytorchBackend(torchPython, 120_000, resolve("ml/pytorch_runner.py"));
    const fromSklearn = await trainedArtifact(sklearn, "logistic_regression", {});
    const fromTorch = await trainedArtifact(torch, "mlp_classifier", CLASSIFICATION.hyperparameters);

    const inTorch = await refusalOf(torch.predict(receiverSnapshot(torch), fromSklearn.model.bytes, ROWS.slice(0, 3), fromSklearn.model.environment, signal()));
    const inSklearn = await refusalOf(sklearn.predict(receiverSnapshot(sklearn), fromTorch.model.bytes, ROWS.slice(0, 3), fromTorch.model.environment, signal()));
    expect(inTorch).toBe("incompatible_artifact");
    expect(inSklearn).toBe("incompatible_artifact");
  }, 600_000);
});

describe.runIf(Boolean(torchPython && tensorflowPython))("PyTorch and TensorFlow refuse each other's artifacts and checkpoints", () => {
  it("refuses an artifact across the two neural frameworks", async () => {
    const torch = new PytorchBackend(torchPython);
    const tensorflow = new TensorflowBackend(tensorflowPython);
    const fromTorch = await trainedArtifact(torch, "mlp_classifier", CLASSIFICATION.hyperparameters);
    const fromTensorflow = await trainedArtifact(tensorflow, "mlp_classifier", CLASSIFICATION.hyperparameters);

    expect(await refusalOf(tensorflow.predict(receiverSnapshot(tensorflow), fromTorch.model.bytes, ROWS.slice(0, 3), fromTorch.model.environment, signal()))).toBe("incompatible_artifact");
    expect(await refusalOf(torch.predict(receiverSnapshot(torch), fromTensorflow.model.bytes, ROWS.slice(0, 3), fromTensorflow.model.environment, signal()))).toBe("incompatible_artifact");
  }, 600_000);

  it("refuses a checkpoint written by the other neural framework, and never resumes it", async () => {
    const torch = new PytorchBackend(torchPython);
    const tensorflow = new TensorflowBackend(tensorflowPython);
    const snapTorch = snapshotWith("pytorch", { algorithm: "mlp_classifier", hyperparameters: { ...quick, learningRate: 0.01 } }, "classification");
    const snapTensorflow = snapshotWith("tensorflow", { algorithm: "mlp_classifier", hyperparameters: { ...quick, learningRate: 0.01 } }, "classification");
    const pTorch = await torch.prepare(snapTorch, ROWS, signal());
    const pTensorflow = await tensorflow.prepare(snapTensorflow, ROWS, signal());
    const torchChunk = await torch.trainChunk!(snapTorch, ROWS, pTorch, null, 1, signal());
    const tfChunk = await tensorflow.trainChunk!(snapTensorflow, ROWS, pTensorflow, null, 1, signal());

    const categories = [
      await refusalOf(tensorflow.trainChunk!(snapTensorflow, ROWS, pTensorflow, torchChunk.checkpoint, 1, signal())),
      await refusalOf(torch.trainChunk!(snapTorch, ROWS, pTorch, tfChunk.checkpoint, 1, signal())),
    ];
    for (const category of categories) expect(["invalid_checkpoint", "incompatible_checkpoint"]).toContain(category);
  }, 600_000);

  it("refuses random bytes as a checkpoint in either framework", async () => {
    const tensorflow = new TensorflowBackend(tensorflowPython);
    const snap = snapshotWith("tensorflow", CLASSIFICATION, "classification");
    const prepared = await tensorflow.prepare(snap, ROWS, signal());
    expect(await refusalOf(tensorflow.trainChunk!(snap, ROWS, prepared, randomBytes(128), 1, signal()))).toBe("invalid_checkpoint");
  }, 600_000);
});

describe.runIf(Boolean(sklearnPython && tensorflowPython))("scikit-learn and TensorFlow refuse each other's artifacts", () => {
  it("refuses a scikit-learn artifact in TensorFlow and a TensorFlow artifact in scikit-learn", async () => {
    const sklearn = new SklearnBackend(sklearnPython);
    const tensorflow = new TensorflowBackend(tensorflowPython);
    const fromSklearn = await trainedArtifact(sklearn, "logistic_regression", {});
    const fromTensorflow = await trainedArtifact(tensorflow, "mlp_classifier", CLASSIFICATION.hyperparameters);

    expect(await refusalOf(tensorflow.predict(receiverSnapshot(tensorflow), fromSklearn.model.bytes, ROWS.slice(0, 3), fromSklearn.model.environment, signal()))).toBe("incompatible_artifact");
    expect(await refusalOf(sklearn.predict(receiverSnapshot(sklearn), fromTensorflow.model.bytes, ROWS.slice(0, 3), fromTensorflow.model.environment, signal()))).toBe("incompatible_artifact");
  }, 600_000);
});

describe.runIf(Boolean(jaxPython && torchPython))("JAX and PyTorch refuse each other's artifacts and checkpoints", () => {
  it("refuses an artifact and a checkpoint across the two frameworks, which register the same model ids", async () => {
    const jax = new JaxBackend(jaxPython);
    const torch = new PytorchBackend(torchPython);
    const fromJax = await trainedArtifact(jax, "mlp_classifier", CLASSIFICATION.hyperparameters);
    const fromTorch = await trainedArtifact(torch, "mlp_classifier", CLASSIFICATION.hyperparameters);
    expect(await refusalOf(torch.predict(receiverSnapshot(torch), fromJax.model.bytes, ROWS.slice(0, 3), fromJax.model.environment, signal()))).toBe("incompatible_artifact");
    expect(await refusalOf(jax.predict(receiverSnapshot(jax), fromTorch.model.bytes, ROWS.slice(0, 3), fromTorch.model.environment, signal()))).toBe("incompatible_artifact");

    const snapJax = snapshotWith("jax", CLASSIFICATION, "classification"), snapTorch = snapshotWith("pytorch", CLASSIFICATION, "classification");
    const pJax = await jax.prepare(snapJax, ROWS, signal()), pTorch = await torch.prepare(snapTorch, ROWS, signal());
    const jaxChunk = await jax.trainChunk!(snapJax, ROWS, pJax, null, 1, signal());
    const torchChunk = await torch.trainChunk!(snapTorch, ROWS, pTorch, null, 1, signal());
    const categories = [await refusalOf(torch.trainChunk!(snapTorch, ROWS, pTorch, jaxChunk.checkpoint, 1, signal())),
      await refusalOf(jax.trainChunk!(snapJax, ROWS, pJax, torchChunk.checkpoint, 1, signal()))];
    for (const category of categories) expect(["invalid_checkpoint", "incompatible_checkpoint"]).toContain(category);
    expect(await refusalOf(jax.trainChunk!(snapJax, ROWS, pJax, randomBytes(128), 1, signal()))).toBe("invalid_checkpoint");
  }, 600_000);
});

describe.runIf(Boolean(sklearnPython && jaxPython))("scikit-learn and JAX refuse each other's artifacts", () => {
  it("refuses a scikit-learn artifact in JAX and a JAX artifact in scikit-learn", async () => {
    const sklearn = new SklearnBackend(sklearnPython);
    const jax = new JaxBackend(jaxPython);
    const fromSklearn = await trainedArtifact(sklearn, "logistic_regression", {});
    const fromJax = await trainedArtifact(jax, "mlp_classifier", CLASSIFICATION.hyperparameters);
    expect(await refusalOf(jax.predict(receiverSnapshot(jax), fromSklearn.model.bytes, ROWS.slice(0, 3), fromSklearn.model.environment, signal()))).toBe("incompatible_artifact");
    expect(await refusalOf(sklearn.predict(receiverSnapshot(sklearn), fromJax.model.bytes, ROWS.slice(0, 3), fromJax.model.environment, signal()))).toBe("incompatible_artifact");
  }, 600_000);
});
