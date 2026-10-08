import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { expect, it } from "vitest";

const python = process.env.ML_TEST_PYTHON;
const violations = ["different_backend", "unsupported_artifact_format", "unsupported_protocol", "unsupported_artifact_schema", "unreadable_artifact_schema", "sklearn_version_differs", "numpy_version_differs", "scipy_version_differs", "joblib_version_differs", "python_minor_differs"];

// Provenance identity (adapter checksum, platform, python patch) is not artifact compatibility.
it.runIf(Boolean(python))("accepts provenance drift and genuine Phase 1 artifacts, and refuses real contract violations", () => {
  const out = JSON.parse(execFileSync(python!, ["-I", resolve("tests/python/check_compat.py")], { encoding: "utf8", env: { ...process.env, OMP_NUM_THREADS: "1" } }).trim().split("\n").pop()!) as Record<string, boolean>;
  const expected = [
    "new_artifact_records_provenance", "new_artifact_embeds_same_environment", "adapter_checksum_drift_tolerated_request", "adapter_checksum_drift_tolerated_embedded", "checksum_drift_keeps_provenance_untouched",
    ...violations.flatMap(v => [`rejects_${v}_request`, `rejects_${v}_embedded`]), "rejects_missing_runtime_versions", "rejects_non_object_environment", "incompatible_environment_checked_before_unpickling",
    "phase1_artifact_is_really_phase1", "phase1_artifact_infers_under_phase2", "phase1_artifact_evaluates_identically_under_phase2", "phase1_provenance_is_untouched",
    "phase1_artifact_still_refused_on_runtime_mismatch", "phase1_artifact_refused_if_schema_claimed_unsupported", "phase1_artifact_with_misleading_protocol_refused",
  ];
  expect(Object.keys(out).sort()).toEqual([...expected].sort());
  for (const name of expected) expect(out[name], name).toBe(true);
}, 120_000);

// CI must never pass by skipping the real backend tests.
it.runIf(process.env.CI === "true")("CI provides the real scikit-learn environment", () => {
  expect(python, "ML_TEST_PYTHON must be set in CI so real backend tests run").toBeTruthy();
  const version = execFileSync(python!, ["-c", "import sklearn; print(sklearn.__version__)"], { encoding: "utf8" }).trim();
  expect(version).toBe(/scikit-learn==(\S+)/.exec(readFileSync("ml/requirements.txt", "utf8"))![1]);
});
