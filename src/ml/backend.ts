import type { ArtifactReference, MetricValue, Row, Snapshot } from "./domain.js";

export interface ArtifactStore {
  put(tenantId: string, bytes: Uint8Array, mediaType: string): Promise<ArtifactReference>;
  get(tenantId: string, reference: ArtifactReference): Promise<Buffer>;
}
export interface PreparedData { indices: { train: number[]; validation: number[]; test: number[] }; environment: Record<string, string> }
export interface TrainedModel { bytes: Buffer; format: string; environment: Record<string, string>; resolvedHyperparameters: Record<string, unknown> }
// Backend methods do computation only. They never claim jobs or mutate durable state.
export interface MlBackend {
  readonly id: string;
  validate(snapshot: Snapshot): void;
  prepare(snapshot: Snapshot, rows: Row[], signal: AbortSignal): Promise<PreparedData>;
  train(snapshot: Snapshot, rows: Row[], prepared: PreparedData, signal: AbortSignal): Promise<TrainedModel>;
  evaluate(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<MetricValue[]>;
  predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]>;
}
