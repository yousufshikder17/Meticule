import { defineModel, registerModels, type ModelDefinition, type ModelInputs, type ParameterSpec } from "./model-catalog.js";

/**
 * MFCC audio models computed with NumPy/SciPy (ml/audio_runner.py, which enforces every bound again). One audio column
 * of bounded WAV clips. `audio_features` is a fixed transform (nothing is fitted); `audio_mfcc_classifier` trains a softmax
 * head on the standardised features.
 */
const frontEnd: Record<string, ParameterSpec> = {
  sampleRate: { kind: "int", min: 8000, max: 48000, default: 16000 },
  nMfcc: { kind: "int", min: 8, max: 40, default: 13 },
  nMels: { kind: "int", min: 16, max: 128, default: 40 },
  frameMs: { kind: "int", min: 10, max: 50, default: 25 },
  hopMs: { kind: "int", min: 5, max: 25, default: 10 },
};
// The hop must not exceed the frame, and the coefficients kept must not exceed the mel bands.
function consistent(p: Record<string, unknown>): string | null {
  if (Number(p.hopMs ?? 10) > Number(p.frameMs ?? 25)) return "hopMs cannot exceed frameMs";
  if (Number(p.nMfcc ?? 13) > Number(p.nMels ?? 40)) return "nMfcc cannot exceed nMels";
  return null;
}
const AUDIO: ModelInputs = { types: ["audio"], min: 1, max: 1 };
const capabilities = (scores: "probability" | "none") => ({ scores, scaleSensitive: false, seeded: false, iterative: false, devices: ["cpu"], mixedPrecision: false } as const);

export const AUDIO_MODELS: readonly ModelDefinition[] = [
  defineModel({ backend: "audio", id: "audio_features", taskType: "embedding", displayName: "MFCC summary embeddings", inputs: AUDIO, learning: "pretrained", parameters: frontEnd, capabilities: capabilities("none"), check: consistent }),
  defineModel({ backend: "audio", id: "audio_mfcc_classifier", taskType: "classification", displayName: "Softmax classifier over MFCC summaries", inputs: AUDIO, learning: "supervised",
    parameters: { ...frontEnd, l2: { kind: "float", min: 0, max: 100, default: 1 }, maxIter: { kind: "int", min: 10, max: 1000, default: 200 } }, capabilities: capabilities("probability"), check: consistent }),
];

registerModels(AUDIO_MODELS);
