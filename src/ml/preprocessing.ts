import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import type { Snapshot } from "./domain.js";

// Operations are interpreted by the sklearn adapter. "numeric" ops run in order over number features,
// "categorical" ops over string/boolean features, "global" ops over the assembled matrix. All are fitted on training rows only.
const none = z.strictObject({});
export const OPERATIONS = {
  impute_median: { scope: "numeric", parameters: none },
  impute_mean: { scope: "numeric", parameters: none },
  standard_scale: { scope: "numeric", parameters: none },
  min_max_scale: { scope: "numeric", parameters: none },
  impute_most_frequent: { scope: "categorical", parameters: none },
  one_hot_encode: { scope: "categorical", parameters: none },
  // Ordinal order is never inferred: every categorical feature needs its explicit category order.
  ordinal_encode: { scope: "categorical", parameters: z.strictObject({ categories: z.record(z.string(), z.array(z.string().max(200)).min(1).max(1000)) }) },
  variance_threshold: { scope: "global", parameters: z.strictObject({ threshold: z.number().min(0).max(1e9).optional() }) },
} as const;
const ENCODERS = ["one_hot_encode", "ordinal_encode"];

export function validatePipeline(snapshot: Snapshot): void {
  const types = new Map(snapshot.dataset.schema.columns.map(c => [c.name, c.type]));
  const categorical = snapshot.pipeline.definition.features.filter(f => types.get(f) !== "number");
  const steps = snapshot.pipeline.definition.steps;
  const seen = new Set<string>();
  for (const step of steps) {
    const op = (OPERATIONS as Record<string, (typeof OPERATIONS)[keyof typeof OPERATIONS]>)[step.operation];
    if (!op) throw new ConflictError(`Unsupported sklearn pipeline operation: ${step.operation}`);
    if (seen.has(step.operation)) throw new ConflictError(`Duplicate pipeline operation: ${step.operation}`);
    seen.add(step.operation);
    const parsed = op.parameters.safeParse(step.parameters);
    if (!parsed.success) throw new ConflictError(`Invalid parameters for ${step.operation}`);
    if (step.operation === "ordinal_encode") {
      const given = (parsed.data as { categories: Record<string, unknown[]> }).categories;
      const missing = categorical.filter(f => !(f in given)); const extra = Object.keys(given).filter(f => !categorical.includes(f));
      if (missing.length || extra.length) throw new ConflictError("ordinal_encode categories must name exactly the categorical features");
    }
  }
  const ops = steps.map(s => s.operation);
  if (seen.has("impute_median") && seen.has("impute_mean")) throw new ConflictError("Choose one numeric imputer");
  if (seen.has("standard_scale") && seen.has("min_max_scale")) throw new ConflictError("Choose one numeric scaler");
  if (ENCODERS.every(e => seen.has(e))) throw new ConflictError("Choose one categorical encoder");
  const encoder = ops.findIndex(o => ENCODERS.includes(o)), imputer = ops.indexOf("impute_most_frequent");
  if (categorical.length && encoder < 0) throw new ConflictError("Categorical features require one_hot_encode or ordinal_encode");
  if (!categorical.length && (encoder >= 0 || imputer >= 0)) throw new ConflictError("Categorical operations require categorical features");
  if (imputer > encoder && encoder >= 0) throw new ConflictError("impute_most_frequent must precede the encoder");
}
