import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterAll, beforeAll } from "vitest";
import type { ColumnType, Row } from "../../src/ml/domain.js";
import { TransformersBackend } from "../../src/ml/transformers-backend.js";
import { IMAGE_ROWS, TEXT_ROWS } from "./modality-rows.js";
import type { Column, TaskTarget } from "./task-suite.js";

// Tiny seeded models built offline by tests/python/hf_fixtures.py into a temporary model root before the suite runs.
// They are random: conformance proves the execution path and contracts, not semantic quality.
export const hfPython = process.env.ML_TEST_TRANSFORMERS_PYTHON ?? "";
const root = mkdtempSync(join(tmpdir(), "meticule-hf-conformance-"));
beforeAll(() => { if (hfPython) execFileSync(hfPython, ["-I", resolve("tests/python/hf_fixtures.py"), root], { encoding: "utf8", maxBuffer: 16_000_000 }); }, 300_000);
afterAll(() => rmSync(root, { recursive: true, force: true }));

export function transformersTarget(algorithm: string, model: string, modality: "text" | "image", labelled: boolean, metrics: string[], extra: Record<string, unknown> = {}): TaskTarget {
  const columns: Column[] = [{ name: "x", type: modality }, ...(labelled ? [{ name: "label", type: "string" as const }] : [])];
  const rows: Row[] = (modality === "text" ? TEXT_ROWS : IMAGE_ROWS).map(r => (labelled ? r : { x: r.x ?? null }));
  return {
    id: "transformers", create: timeoutMs => new TransformersBackend(hfPython, root, timeoutMs), python: hfPython, script: resolve("ml/transformers_runner.py"),
    algorithm, hyperparameters: { model, ...extra }, columns, feature: "x", target: labelled ? "label" : null, rows, metrics,
    requestExtras: { modelRoot: root }, requiredInCi: process.env.ML_TRANSFORMERS_JOB === "true",
  };
}
