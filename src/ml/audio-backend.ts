import { resolve } from "node:path";
import "./audio-models.js"; // registers the audio definitions
import type { BackendCapabilities } from "./backend.js";
import { RunnerBackend } from "./runner-backend.js";

/** NumPy/SciPy audio on CPU, one fit (or none) per attempt. Runs on the scikit-learn interpreter (requirements.txt). */
export class AudioBackend extends RunnerBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/audio_runner.py")) {
    const capabilities: BackendCapabilities = { artifactFormats: ["audio-npz-v1"], checkpointFormats: [], devices: ["cpu"], precisions: ["fp32"], preprocessing: [], iterative: false, tasks: ["embedding", "classification"], inputs: ["audio"] };
    super({ id: "audio", capabilities, python, script, timeoutMs });
  }
}
