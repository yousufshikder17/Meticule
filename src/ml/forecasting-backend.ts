import { resolve } from "node:path";
import type { BackendCapabilities } from "./backend.js";
import "./forecasting-models.js"; // registers the forecasting definitions
import { RunnerBackend } from "./runner-backend.js";

/** Univariate forecasting on CPU (pandas, NumPy, scikit-learn), one fit per attempt. Runs on the scikit-learn interpreter. */
export class ForecastingBackend extends RunnerBackend {
  constructor(python: string, timeoutMs = 120_000, script = resolve(process.cwd(), "ml/forecasting_runner.py")) {
    const capabilities: BackendCapabilities = { artifactFormats: ["forecast-npz-v1"], checkpointFormats: [], devices: ["cpu"], precisions: ["fp32"], preprocessing: [], iterative: false, tasks: ["forecasting"], inputs: ["timestamp"] };
    super({ id: "forecasting", capabilities, python, script, timeoutMs });
  }
}
