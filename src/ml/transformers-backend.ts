import { resolve } from "node:path";
import { ConflictError } from "../domain/errors.js";
import type { BackendCapabilities } from "./backend.js";
import type { Snapshot } from "./domain.js";
import { RunnerBackend } from "./runner-backend.js";
import "./transformers-models.js"; // registers the transformers definitions

export const TRANSFORMERS_ARTIFACT_FORMAT = "transformers-reference-v1";

/**
 * Hugging Face Transformers on CPU over provisioned local models. The runner verifies the model directory against its
 * manifest on every command and loads only a private copy; it never downloads anything and never runs remote code. The
 * artifact references the model by name and verified digest and holds any trained head, never the base weights.
 * The model root is platform-wide: every tenant may use every provisioned model. It is handed to the runner per request and
 * never recorded in environments, artifacts or errors.
 */
export class TransformersBackend extends RunnerBackend {
  constructor(python: string, private readonly modelRoot: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/transformers_runner.py")) {
    const capabilities: BackendCapabilities = { artifactFormats: [TRANSFORMERS_ARTIFACT_FORMAT], checkpointFormats: [], devices: ["cpu"], precisions: ["fp32"], preprocessing: [], iterative: false, tasks: ["classification", "embedding", "transcription"], inputs: ["text", "image", "audio"] };
    super({ id: "transformers", capabilities, python, script, timeoutMs });
  }
  override validate(snapshot: Snapshot): void {
    super.validate(snapshot);
    if (typeof snapshot.hyperparameters.model !== "string") throw new ConflictError("Transformers models need the name of a provisioned model");
  }
  protected override requestExtras(): Record<string, unknown> { return { modelRoot: this.modelRoot }; }
}
