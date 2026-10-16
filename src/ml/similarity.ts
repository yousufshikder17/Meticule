import { ConflictError } from "../domain/errors.js";

/**
 * Vector similarity over finite number arrays. cosine and dot are similarities (higher is closer); euclidean is a distance
 * (lower is closer). Malformed input is refused, never turned into NaN: empty vectors, mismatched dimensions, non-finite
 * values, and (for cosine) zero vectors. pgvector computes the same three measures in SQL (`<=>`, `<#>`, `<->`).
 */
export const SIMILARITY_METRICS = ["cosine", "dot", "euclidean"] as const;
export type SimilarityMetric = (typeof SIMILARITY_METRICS)[number];
export const HIGHER_IS_CLOSER: Record<SimilarityMetric, boolean> = { cosine: true, dot: true, euclidean: false };

function checked(vector: readonly number[], dimension?: number): readonly number[] {
  if (!vector.length) throw new ConflictError("Vectors must not be empty");
  if (dimension !== undefined && vector.length !== dimension) throw new ConflictError("Vector dimensions differ");
  if (!vector.every(Number.isFinite)) throw new ConflictError("Vectors must contain only finite numbers");
  return vector;
}

function norm(vector: readonly number[]): number {
  return Math.sqrt(vector.reduce((sum, x) => sum + x * x, 0));
}

/** The metric's value for one pair; cosine refuses a zero vector instead of dividing by zero. */
export function similarity(metric: SimilarityMetric, a: readonly number[], b: readonly number[]): number {
  checked(a); checked(b, a.length);
  if (metric === "euclidean") return Math.sqrt(a.reduce((sum, x, i) => sum + (x - b[i]!) ** 2, 0));
  const dot = a.reduce((sum, x, i) => sum + x * b[i]!, 0);
  if (metric === "dot") return dot;
  const na = norm(a), nb = norm(b);
  if (na === 0 || nb === 0) throw new ConflictError("Cosine similarity is undefined for a zero vector");
  return Math.min(1, Math.max(-1, dot / (na * nb)));
}

/** Every candidate scored against the query, closest first; ties keep the lower index first, so the order is deterministic. */
export function rank(metric: SimilarityMetric, query: readonly number[], candidates: readonly (readonly number[])[]): { index: number; score: number }[] {
  const scored = candidates.map((candidate, index) => ({ index, score: similarity(metric, query, candidate) }));
  const sign = HIGHER_IS_CLOSER[metric] ? -1 : 1;
  return scored.sort((x, y) => sign * (x.score - y.score) || x.index - y.index);
}

/** left x right matrix of the metric. */
export function similarityMatrix(metric: SimilarityMetric, left: readonly (readonly number[])[], right: readonly (readonly number[])[]): number[][] {
  return left.map(a => right.map(b => similarity(metric, a, b)));
}
