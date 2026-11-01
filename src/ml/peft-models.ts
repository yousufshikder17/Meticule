import { defineModel, registerModels, type ModelDefinition } from "./model-catalog.js";

/**
 * Phase 6B.2: LoRA fine-tuning with Hugging Face PEFT on one bounded base model, `meticule-tiny-llama-v1`, which the runner
 * builds from a fixed configuration and seed, or a bounded operator-provisioned Llama selected by manifest name.
 * Next-token language modelling over one `tokens` column, CPU/CUDA fp32 adapters with optional four-bit CUDA base weights.
 * Mirrors LIMITS in ml/lora_backend.py, which also bounds what only the data
 * shows: vocabulary ids, sequence lengths, optimizer steps and training tokens.
 */
export const PEFT_DEVICES = ["cpu", "cuda"] as const;
export const LORA_LIMITS = { maxRows: 1024, maxRank: 16, maxAlpha: 64, maxDropout: 0.5, maxLearningRate: 0.1, maxBatchSize: 32, maxEpochs: 100, maxSequenceLength: 32 } as const;
export const LORA_BASE_MODELS = ["meticule-tiny-llama-v1", "provisioned-llama"] as const;
export const LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"] as const;

const LORA_CAUSAL_LM = defineModel({
  backend: "peft", id: "lora_causal_lm", taskType: "language_modeling", displayName: "LoRA fine-tuning of a causal language model (PEFT)",
  inputs: { types: ["tokens"], min: 1, max: 1 },
  parameters: {
    baseModel: { kind: "choice", values: LORA_BASE_MODELS, default: "meticule-tiny-llama-v1" },
    pretrainedModel: { kind: "name", default: null },
    rank: { kind: "int", min: 1, max: LORA_LIMITS.maxRank, default: 4 },
    alpha: { kind: "float", min: 0, max: LORA_LIMITS.maxAlpha, openMin: true, default: 8 },
    dropout: { kind: "float", min: 0, max: LORA_LIMITS.maxDropout, default: 0 },
    targetModules: { kind: "choiceList", values: LORA_TARGET_MODULES, minItems: 1, maxItems: LORA_TARGET_MODULES.length, default: ["q_proj", "v_proj"] },
    learningRate: { kind: "float", min: 0, max: LORA_LIMITS.maxLearningRate, openMin: true, default: 0.02 },
    batchSize: { kind: "int", min: 1, max: LORA_LIMITS.maxBatchSize, default: 16 },
    epochs: { kind: "int", min: 1, max: LORA_LIMITS.maxEpochs, default: 10 },
    maxSequenceLength: { kind: "int", min: 2, max: LORA_LIMITS.maxSequenceLength, default: 32 },
    quantization: { kind: "choice", values: ["none", "nf4", "fp4"], default: "none" },
    computeDtype: { kind: "choice", values: ["fp32", "bf16"], default: "fp32" },
    doubleQuantization: { kind: "bool", default: false },
    maxMemoryMiB: { kind: "int", min: 256, max: 65536, default: 6144 },
    gradientAccumulationSteps: { kind: "int", min: 1, max: 32, default: 1 },
  },
  check: p => {
    if ((p.quantization ?? "none") === "none" && ((p.computeDtype ?? "fp32") !== "fp32" || p.doubleQuantization === true))
      return "Compute dtype and double quantization require four-bit quantization";
    if (p.baseModel === "provisioned-llama" ? typeof p.pretrainedModel !== "string" : p.pretrainedModel != null)
      return "A provisioned Llama base requires exactly one provisioned model name";
    return null;
  },
  capabilities: { scores: "none", scaleSensitive: false, seeded: true, iterative: true, devices: PEFT_DEVICES, mixedPrecision: false },
});

export const PEFT_MODELS: readonly ModelDefinition[] = [LORA_CAUSAL_LM];

registerModels(PEFT_MODELS);
