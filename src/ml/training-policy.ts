import { ConflictError } from "../domain/errors.js";
import { getModel } from "./model-catalog.js";
import type { Snapshot } from "./domain.js";
import { z } from "zod";

export interface EffectivePolicy { checkpointEveryEpochs: number; earlyStopping: { patience: number; minDelta: number } | null; device: "cpu" | "cuda"; precision: "fp32" | "mixed" }

export type EpochSummary = { epoch: number; trainLoss: number; validationLoss: number | null; elapsedSeconds: number };
export const EpochSummarySchema = z.object({ epoch: z.int().min(1).max(1000), trainLoss: z.number().finite(), validationLoss: z.number().finite().nullable(), elapsedSeconds: z.number().finite().nonnegative() });

/** Epochs the snapshot requests. Catalog defaults fill the same role they do for model hyperparameters. */
export function requestedEpochs(snapshot: Snapshot): number {
  const model = getModel(snapshot.backend, snapshot.algorithm);
  const epochs = { ...model.defaults, ...snapshot.hyperparameters }.epochs;
  if (typeof epochs !== "number") throw new ConflictError("Iterative model has no epoch count");
  return epochs;
}

/** Iterative training is a capability of the model definition, never a backend name check. */
export function isIterative(snapshot: Snapshot): boolean {
  return getModel(snapshot.backend, snapshot.algorithm).capabilities.iterative === true;
}

/**
 * Validates the policy against the model. Early stopping needs an iterative model with a validation partition, and
 * it forces one epoch per chunk, so a stop decision never trains past the epoch where it was made.
 */
export function effectivePolicy(snapshot: Snapshot): EffectivePolicy {
  const policy = snapshot.training;
  if (!policy) return { checkpointEveryEpochs: 5, earlyStopping: null, device: "cpu", precision: "fp32" };
  if (policy.earlyStopping && !isIterative(snapshot)) throw new ConflictError("Early stopping requires an iterative model");
  if (policy.earlyStopping && snapshot.split.validation <= 0) throw new ConflictError("Early stopping requires a validation partition");
  return {
    checkpointEveryEpochs: policy.earlyStopping ? 1 : policy.checkpointEveryEpochs,
    earlyStopping: policy.earlyStopping,
    device: policy.device,
    precision: policy.precision,
  };
}

/**
 * Early stopping on validation loss. "Best" is the strict minimum, so the best epoch is the same whether it is
 * tracked here or in the backend. `minDelta` only decides whether an epoch counts as progress for patience.
 */
export class EarlyStopper {
  private best: { epoch: number; value: number } | null = null;
  private sinceProgress = 0;
  private stopped: { epoch: number } | null = null;
  constructor(private readonly patience: number, private readonly minDelta: number) {}
  observe(summary: { epoch: number; validationLoss: number | null }): void {
    if (summary.validationLoss === null) throw new ConflictError("Early stopping needs a validation loss for every epoch");
    const value = summary.validationLoss;
    const previous = this.best;
    if (!previous || value < previous.value) this.best = { epoch: summary.epoch, value };
    if (!previous || previous.value - value > this.minDelta) this.sinceProgress = 0;
    else this.sinceProgress++;
    if (!this.stopped && this.sinceProgress >= this.patience) this.stopped = { epoch: summary.epoch };
  }
  get shouldStop(): boolean { return this.stopped !== null; }
  get bestEpoch(): number | null { return this.best?.epoch ?? null; }
  get bestValue(): number | null { return this.best?.value ?? null; }
  /** Rebuilds the stopper from the epoch summaries already persisted, so a resumed attempt decides identically. */
  static replay(policy: { patience: number; minDelta: number }, summaries: { epoch: number; validationLoss: number | null }[]): EarlyStopper {
    const stopper = new EarlyStopper(policy.patience, policy.minDelta);
    for (const s of summaries) stopper.observe(s);
    return stopper;
  }
}
