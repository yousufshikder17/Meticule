import { resolve } from "node:path";
import type { Row } from "../../src/ml/domain.js";
import { AudioBackend } from "../../src/ml/audio-backend.js";
import { AUDIO_ROWS } from "./modality-rows.js";
import type { Column, TaskTarget } from "./task-suite.js";

// The audio backend runs on the scikit-learn interpreter; CI's main test job provides it as ML_TEST_AUDIO_PYTHON.
export const audioPython = process.env.ML_TEST_AUDIO_PYTHON ?? "";

export function audioTarget(algorithm: string, labelled: boolean, metrics: string[], hyperparameters: Record<string, unknown> = {}): TaskTarget {
  const columns: Column[] = [{ name: "x", type: "audio" }, ...(labelled ? [{ name: "label", type: "string" as const }] : [])];
  const rows: Row[] = AUDIO_ROWS.map(r => (labelled ? r : { x: r.x ?? null }));
  return {
    id: "audio", create: timeoutMs => new AudioBackend(audioPython, timeoutMs), python: audioPython, script: resolve("ml/audio_runner.py"),
    algorithm, hyperparameters, columns, feature: "x", target: labelled ? "label" : null, rows, metrics, requiredInCi: process.env.CI === "true",
  };
}
