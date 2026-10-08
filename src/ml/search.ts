import { z } from "zod";
import { ConflictError } from "../domain/errors.js";
import { canonicalJson } from "../domain/canonical-json.js";
import { validateHyperparameters, type ModelDefinition } from "./model-catalog.js";

type Scalar = string | number | boolean | null;
export type Point = Record<string, Scalar>;
const ScalarValue = z.union([z.string().max(100), z.number().finite(), z.boolean(), z.null()]);
export const Dimension = z.discriminatedUnion("type", [
  z.strictObject({ type: z.literal("choice"), values: z.array(ScalarValue).min(1).max(100) }),
  z.strictObject({ type: z.literal("int"), min: z.int(), max: z.int(), step: z.int().min(1).optional() }),
  z.strictObject({ type: z.literal("float"), min: z.number().finite(), max: z.number().finite(), log: z.boolean().optional(), steps: z.int().min(2).max(20).optional() }),
]);
export type DimensionType = z.infer<typeof Dimension>;
export const SearchSpace = z.record(z.string().min(1).max(60), Dimension).refine(s => Object.keys(s).length >= 1, "Search space is empty").refine(s => Object.keys(s).length <= 10, "At most 10 searched parameters");
export type SearchSpaceType = z.infer<typeof SearchSpace>;
const GRID_LIMIT = 10_000;

/** Framework-neutral: strategies see only validated dimensions, never a backend or an estimator. */
export interface SearchStrategy { readonly id: string; generate(space: SearchSpaceType, accepts: (point: Point) => boolean, maxCandidates: number, seed: number): Point[] }

const round = (v: number) => Number(v.toPrecision(12));
const intValues = (d: Extract<DimensionType, { type: "int" }>) => Array.from({ length: Math.floor((d.max - d.min) / (d.step ?? 1)) + 1 }, (_, i) => d.min + i * (d.step ?? 1));
const floatValues = (d: Extract<DimensionType, { type: "float" }>): number[] => {
  const n = d.steps!; if (d.min === d.max) return [d.min];
  return Array.from({ length: n }, (_, i) => i === 0 ? d.min : i === n - 1 ? d.max : round(d.log ? Math.exp(Math.log(d.min) + (Math.log(d.max) - Math.log(d.min)) * i / (n - 1)) : d.min + (d.max - d.min) * i / (n - 1)));
};
const finiteValues = (d: DimensionType): Scalar[] => d.type === "choice" ? d.values : d.type === "int" ? intValues(d) : floatValues(d);
const names = (space: SearchSpaceType) => Object.keys(space).sort();

export const GridSearch: SearchStrategy = {
  id: "grid",
  /** Cartesian product, parameters sorted by name, values in declared order, last parameter varying fastest. */
  generate(space, accepts, maxCandidates) {
    const keys = names(space); const axes = keys.map(k => finiteValues(space[k]!));
    const total = axes.reduce((n, a) => n * a.length, 1);
    if (total > GRID_LIMIT) throw new ConflictError(`Grid has ${total} combinations; the limit is ${GRID_LIMIT}`);
    const points: Point[] = []; const walk = (depth: number, current: Point) => {
      if (depth === keys.length) { if (accepts(current)) points.push(current); return; }
      for (const value of axes[depth]!) walk(depth + 1, { ...current, [keys[depth]!]: value });
    };
    walk(0, {});
    if (points.length > maxCandidates) throw new ConflictError(`Grid yields ${points.length} valid candidates, above maxCandidates=${maxCandidates}; narrow the space or use random search`);
    return points;
  },
};

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => { a = (a + 0x6D2B79F5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export const RandomSearch: SearchStrategy = {
  id: "random",
  /** Seeded sampling without replacement; stops early when the finite space is exhausted. */
  generate(space, accepts, maxCandidates, seed) {
    const keys = names(space); const next = mulberry32(seed);
    const draw = (d: DimensionType): Scalar => {
      if (d.type === "choice") return d.values[Math.floor(next() * d.values.length)]!;
      if (d.type === "int") { const v = intValues(d); return v[Math.floor(next() * v.length)]!; }
      if (d.min === d.max) return d.min;
      const raw = d.log ? Math.exp(Math.log(d.min) + (Math.log(d.max) - Math.log(d.min)) * next()) : d.min + (d.max - d.min) * next();
      return Math.min(d.max, Math.max(d.min, Number(raw.toPrecision(6))));
    };
    const unbounded = keys.some(k => { const d = space[k]!; return d.type === "float" && d.min !== d.max; });
    const capacity = unbounded ? Infinity : keys.reduce((n, k) => n * finiteValues(space[k]!).length, 1);
    const seen = new Set<string>(); const points: Point[] = [];
    for (let attempts = 0; points.length < maxCandidates && attempts < maxCandidates * 100 + 100 && seen.size < capacity; attempts++) {
      const point = Object.fromEntries(keys.map(k => [k, draw(space[k]!)])) as Point; const key = canonicalJson(point);
      if (seen.has(key)) continue; seen.add(key);
      if (accepts(point)) points.push(point);
    }
    return points;
  },
};
export const STRATEGIES: Record<string, SearchStrategy> = { grid: GridSearch, random: RandomSearch };

/** Rejects anything outside the model's allowlist before a single candidate is generated. */
export function validateSearchSpace(model: ModelDefinition, space: SearchSpaceType, fixed: Record<string, unknown>): void {
  validateHyperparameters(model, fixed);
  for (const [name, d] of Object.entries(space)) {
    if (!(name in model.parameters)) throw new ConflictError(`${model.id}: unsupported search parameter ${name}`);
    if (name in fixed) throw new ConflictError(`${model.id}: ${name} is both fixed and searched`);
    const check = (value: Scalar) => { if (!model.parameterSchema.safeParse({ [name]: value }).success) throw new ConflictError(`${model.id}: invalid value for ${name}`); };
    if (d.type === "choice") {
      if (new Set(d.values.map(v => canonicalJson(v))).size !== d.values.length) throw new ConflictError(`${name}: duplicate choices`);
      d.values.forEach(check);
    } else {
      if (d.min > d.max) throw new ConflictError(`${name}: min exceeds max`);
      if (d.type === "int") { if (d.step !== undefined && d.step > d.max - d.min && d.max !== d.min) throw new ConflictError(`${name}: step exceeds the range`); if ((d.max - d.min) / (d.step ?? 1) > 1000) throw new ConflictError(`${name}: range has too many values`); }
      else if (d.log && d.min <= 0) throw new ConflictError(`${name}: log scale needs min > 0`);
      check(d.min); check(d.max);
    }
  }
}
export function gridNeedsSteps(space: SearchSpaceType, strategy: string): void {
  if (strategy !== "grid") return;
  for (const [name, d] of Object.entries(space)) if (d.type === "float" && d.min !== d.max && d.steps === undefined) throw new ConflictError(`${name}: grid search needs steps for a float range`);
}

export function generateCandidates(model: ModelDefinition, space: SearchSpaceType, fixed: Record<string, unknown>, strategyId: string, maxCandidates: number, seed: number): Point[] {
  const strategy = STRATEGIES[strategyId]; if (!strategy) throw new ConflictError(`Unsupported search strategy: ${strategyId}`);
  validateSearchSpace(model, space, fixed); gridNeedsSteps(space, strategyId);
  // Combinations the model rejects (for example degree with a non-poly kernel) are dropped, not sent to a worker.
  const accepts = (point: Point) => { try { validateHyperparameters(model, { ...fixed, ...point }); return true; } catch { return false; } };
  const points = strategy.generate(space, accepts, maxCandidates, seed);
  if (!points.length) throw new ConflictError("No valid hyperparameter combination exists in the search space");
  return points;
}
