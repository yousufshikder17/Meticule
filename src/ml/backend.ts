import type { ArtifactReference, MetricValue, Row, Snapshot } from "./domain.js";

export interface ArtifactStore {
  put(tenantId: string, bytes: Uint8Array, mediaType: string): Promise<ArtifactReference>;
  get(tenantId: string, reference: ArtifactReference): Promise<Buffer>;
}
/** `folds` holds each cross-validation fold's validation indices (a partition of `indices.train`) when the spec asks for CV. */
export interface PreparedData { indices: { train: number[]; validation: number[]; test: number[] }; folds?: number[][] | undefined; environment: Record<string, string> }
/** Measured seconds reported by the backend; absent keys mean "not measured", never zero-filled. */
export type Performance = Record<string, number>;
export interface TrainedModel { bytes: Buffer; format: string; environment: Record<string, string>; resolvedHyperparameters: Record<string, unknown>; performance?: Performance | undefined }
export interface Evaluation { metrics: MetricValue[]; performance?: Performance | undefined }
// Backend methods do computation only. They never claim jobs or mutate durable state.
export interface MlBackend {
  readonly id: string;
  /** Artifact formats this backend can load; inference refuses any other stored format before touching the bytes. */
  readonly artifactFormats?: readonly string[] | undefined;
  validate(snapshot: Snapshot): void;
  prepare(snapshot: Snapshot, rows: Row[], signal: AbortSignal): Promise<PreparedData>;
  train(snapshot: Snapshot, rows: Row[], prepared: PreparedData, signal: AbortSignal): Promise<TrainedModel>;
  evaluate(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<MetricValue[]>;
  /** Optional richer form that also reports measured timings; the processor prefers it when present. */
  evaluateDetailed?(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<Evaluation>;
  predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]>;
}
