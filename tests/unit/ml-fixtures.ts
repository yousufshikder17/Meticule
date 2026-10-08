import { randomUUID } from "node:crypto";
import { TrainingSnapshot } from "../../src/ml/domain.js";

type Column = { name: string; type: "number" | "string" | "boolean"; nullable?: boolean };
/** Builds a validated snapshot without a database, for catalog/preprocessing/search tests. */
export function snapshotFor(opts: { algorithm?: string; columns?: Column[]; target?: string; features?: string[]; steps?: { operation: string; parameters?: Record<string, unknown> }[]; hyperparameters?: Record<string, unknown>; evaluation?: object; stratify?: boolean; rowCount?: number } = {}) {
  const id = randomUUID(), tenantId = randomUUID();
  const columns = opts.columns ?? [{ name: "x", type: "number" as const }, { name: "y", type: "number" as const }];
  return TrainingSnapshot.parse({
    datasetVersionId: id, featurePipelineId: id, backend: "sklearn", algorithm: opts.algorithm ?? "ridge", seed: 42, hyperparameters: opts.hyperparameters ?? {}, evaluation: opts.evaluation ?? {},
    split: { id: randomUUID(), strategy: "random", train: .6, validation: .2, test: .2, stratify: opts.stratify ?? false },
    dataset: { id, tenantId, createdAt: new Date().toISOString(), datasetId: id, version: 1, schema: { id: "11111111-1111-4111-8111-111111111111", columns, target: opts.target ?? "y" }, content: { key: "a".repeat(64), sha256: "a".repeat(64), size: 1, mediaType: "application/json" }, rowCount: opts.rowCount ?? 100, validation: { valid: true } },
    pipeline: { id, tenantId, logicalId: id, version: 1, name: "fixture", createdAt: new Date().toISOString(), definition: { features: opts.features ?? ["x"], steps: opts.steps ?? [] } },
  });
}
