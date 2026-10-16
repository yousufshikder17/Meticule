import { defineModel, registerModels, type ModelDefinition, type ModelInputs, type ParameterSpec } from "./model-catalog.js";

/**
 * Pretrained Hugging Face models provisioned under the platform-wide model root (ML_TRANSFORMERS_MODEL_ROOT), referenced by
 * the required `model` name. Mirrors ml/transformers_runner.py, which enforces every bound again. CPU, fp32, one fit
 * (a trained head) or none (pretrained) per attempt.
 */
const model: ParameterSpec = { kind: "name", default: null };
const batchSize: ParameterSpec = { kind: "int", min: 1, max: 64, default: 16 };
const maxLength: ParameterSpec = { kind: "int", min: 8, max: 512, default: 128 };
const head: Record<string, ParameterSpec> = { l2: { kind: "float", min: 0, max: 100, default: 1 }, maxIter: { kind: "int", min: 10, max: 1000, default: 200 } };
const textPooling: ParameterSpec = { kind: "choice", values: ["mean", "cls"], default: "mean" };
const imagePooling: ParameterSpec = { kind: "choice", values: ["cls", "mean"], default: "cls" };
const normalize: ParameterSpec = { kind: "bool", default: true };
const TEXT: ModelInputs = { types: ["text"], min: 1, max: 1 };
const IMAGE: ModelInputs = { types: ["image"], min: 1, max: 1 };
const AUDIO: ModelInputs = { types: ["audio"], min: 1, max: 1 };
const capabilities = (scores: "probability" | "none") => ({ scores, scaleSensitive: false, seeded: false, iterative: false, devices: ["cpu"], mixedPrecision: false } as const);
const define = (id: string, taskType: "classification" | "embedding" | "transcription", displayName: string, inputs: ModelInputs, learning: "pretrained" | "supervised", parameters: Record<string, ParameterSpec>) =>
  defineModel({ backend: "transformers", id, taskType, displayName, inputs, learning, parameters, capabilities: capabilities(taskType === "classification" ? "probability" : "none") });

export const TRANSFORMERS_MODELS: readonly ModelDefinition[] = [
  define("text_classifier", "classification", "Pretrained text classifier (BERT/DistilBERT)", TEXT, "pretrained", { model, maxLength, batchSize }),
  define("text_embedder", "embedding", "Text embeddings (BERT/DistilBERT)", TEXT, "pretrained", { model, maxLength, batchSize, pooling: textPooling, normalize }),
  define("text_embedding_classifier", "classification", "Softmax head over a frozen text encoder", TEXT, "supervised", { model, maxLength, batchSize, pooling: textPooling, ...head }),
  define("image_classifier", "classification", "Pretrained image classifier (ViT)", IMAGE, "pretrained", { model, batchSize }),
  define("image_embedder", "embedding", "Image embeddings (ViT)", IMAGE, "pretrained", { model, batchSize, pooling: imagePooling, normalize }),
  define("image_embedding_classifier", "classification", "Softmax head over a frozen image encoder", IMAGE, "supervised", { model, batchSize, pooling: imagePooling, ...head }),
  // Audio (wav2vec2) is forwarded one clip at a time, so no batch size; speech decodes greedily (CTC).
  define("audio_classifier", "classification", "Pretrained audio classifier (wav2vec2)", AUDIO, "pretrained", { model }),
  define("speech_recognizer", "transcription", "Speech recognition (wav2vec2 CTC, greedy)", AUDIO, "pretrained", { model }),
];

registerModels(TRANSFORMERS_MODELS);
