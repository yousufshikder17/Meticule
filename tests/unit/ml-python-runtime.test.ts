import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { METRICS, defaultMetrics, type MetricName } from "../../src/ml/metrics.js";
import { TASK_TYPES } from "../../src/ml/tasks.js";

const python = process.env.ML_TEST_PYTHON;

/** Runs one tests/python check with the sklearn interpreter and returns its last stdout line, parsed. */
function check<T extends { results: Record<string, boolean> }>(script: string): T {
  const out = execFileSync(python!, ["-I", "-B", resolve("tests/python", script)], { encoding: "utf8", maxBuffer: 16_000_000, env: { ...process.env, OMP_NUM_THREADS: "1" } });
  return JSON.parse(out.trim().split("\n").pop()!) as T;
}

/** Every named check present (a dropped one fails the suite rather than passing quietly) and every one true. */
function expectAll(results: Record<string, boolean>, expected: string[]) {
  expect(Object.keys(results).sort()).toEqual([...expected].sort());
  for (const name of expected) expect(results[name], name).toBe(true);
}

// The shared Python runtime behind every runner: contract envelope (A1/A3/A5), split partitions (sec. 3), metrics (sec. 7, D1/D2)
// and media/timestamp validation (C1-C4) against the vector file that src/ml/media.ts must also satisfy.
describe.runIf(Boolean(python))("Python runtime common", () => {
  it("serves contract v1 and v2 with the semantic check before execute and a stdout only the response can reach", () => {
    expectAll(check("check_contract.py").results, ["v0_refused_before_any_work", "missing_or_malformed_envelope_refused", "v1_tabular_request_runs_and_echoes_v1",
      "v2_request_runs_and_echoes_v2", "v1_extended_prepare_refused", "v1_extended_predict_refused", "v1_phase4_task_refused", "v1_unknown_algorithm_refused",
      "v2_task_mismatch_refused", "v2_unknown_algorithm_refused", "v2_missing_task_refused", "catalog_skips_semantic_check", "request_without_snapshot_reaches_execute",
      "served_without_task_of_speaks_v1_only", "refusal_messages_are_fixed", "uses_extensions_matches_contract_table", "python_stdout_goes_to_stderr_with_custom_streams",
      "process_stdout_carries_only_the_response", "refusal_is_last_stderr_line_after_native_noise"]);
  }, 180_000);

  it("keeps the random split byte-identical and splits chronologically by UTC instant with floor sizes", () => {
    expectAll(check("check_partitions.py").results, ["random_path_matches_phase3_partitions", "unlabelled_random_splits_like_zero_targets",
      "unlabelled_random_refuses_stratification", "chronological_sizes_floor_ieee_products", "chronological_blocks_follow_time_order",
      "chronological_partitions_cover_every_row_once", "chronological_orders_by_utc_instant_not_text", "chronological_ties_keep_row_order",
      "chronological_split_ignores_seed", "chronological_refuses_empty_partition", "chronological_refuses_invalid_timestamps",
      "chronological_refuses_cross_validation", "unknown_strategy_refused", "sklearn_prepare_uses_split_partitions"]);
  }, 120_000);

  it("computes every Phase 4 metric as hand-computed, refuses undefined values, and dumps the registry mirror", () => {
    const out = check<{ results: Record<string, boolean>; registry: Record<string, unknown> }>("check_metrics.py");
    expectAll(out.results, ["registry_matches_phase4_spec", "phase3_tables_unchanged", "derived_tables_consistent", "default_metrics_per_task",
      "requested_metrics_rules", "existing_metric_values_unchanged", "smape_hand_computed", "smape_zero_terms_and_range", "mase_hand_computed",
      "mase_refuses_zero_or_missing_scale", "average_precision_hand_computed", "anomaly_roc_auc_hand_computed", "anomaly_scores_hand_computed",
      "anomaly_zero_division_is_zero", "binary_metrics_refuse_single_class", "threshold_calibration_error_hand_computed", "retrieval_hand_computed",
      "retrieval_is_scale_invariant", "retrieval_excludes_rows_without_a_peer", "retrieval_ties_rank_non_relevant_first", "retrieval_blockwise_equals_single_block",
      "retrieval_refuses_undefined", "mean_pairwise_cosine_hand_computed", "mean_pairwise_cosine_matches_all_pairs", "mean_pairwise_cosine_refuses_undefined",
      "levenshtein_matches_dynamic_programming", "wer_cer_hand_computed", "wer_cer_refuse_empty_references"]);
    // The mirror must equal the TypeScript registry: names, directions, task membership, label need, diagnostics, defaults.
    const names = Object.keys(METRICS) as MetricName[];
    const unlabelled = (task: (typeof TASK_TYPES)[number]) => { try { return defaultMetrics(task, false); } catch { return null; } };
    expect(out.registry).toEqual({
      directions: Object.fromEntries(names.map(name => [name, METRICS[name].direction])),
      taskMetrics: Object.fromEntries(TASK_TYPES.map(task => [task, names.filter(name => METRICS[name].tasks.includes(task)).sort()])),
      labelFree: names.filter(name => !METRICS[name].labels).sort(),
      diagnostic: names.filter(name => METRICS[name].diagnostic).sort(),
      defaults: Object.fromEntries(TASK_TYPES.map(task => [task, { labelled: defaultMetrics(task, true), unlabelled: unlabelled(task) }])),
    });
  }, 120_000);

  it("decides every shared media and timestamp vector as the vector file says", () => {
    const out = check<{ results: Record<string, boolean>; vectors: Record<string, boolean>; mismatches: string[] }>("check_media.py");
    expect(out.mismatches).toEqual([]);
    expectAll(out.results, ["vector_file_well_formed", "limits_match_media_common", "every_attack_case_present", "base64_vectors_agree", "image_vectors_agree",
      "audio_vectors_agree", "timestamp_vectors_agree", "media_refusals_are_invalid_media", "timestamp_refusals_are_invalid_time_series",
      "refusal_messages_are_fixed", "wav_decoder_scales_each_width", "wav_decoder_returns_frames_by_channels", "wav_decoder_accepts_extensible",
      "parse_timestamp_is_utc_epoch_seconds", "epoch_micros_is_exact"]);
    const ids = (JSON.parse(readFileSync(resolve("tests/fixtures/media-vectors.json"), "utf8")) as { vectors: { id: string }[] }).vectors.map(v => v.id);
    expectAll(out.vectors, ids);
  }, 120_000);
});
