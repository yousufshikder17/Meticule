import { DEVICES, PRECISIONS, type ArtifactReference, type MetricValue, type Row, type Snapshot } from "./domain.js";

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
export type Device = (typeof DEVICES)[number];
export type Precision = (typeof PRECISIONS)[number];
/**
 * What a backend's runner can execute, declared once by the backend. Meticule checks requests against it before any
 * work starts, and a model's own capabilities must fit inside it (the conformance suite enforces that).
 */
export interface BackendCapabilities {
  /** Artifact formats the runner can load. Inference refuses any other stored format before touching the bytes. */
  readonly artifactFormats: readonly string[];
  /** Checkpoint formats the runner can resume. Empty for one-shot backends. */
  readonly checkpointFormats: readonly string[];
  /** Preprocessing operations the runner interprets. Any other pipeline step is refused before a runner starts. */
  readonly preprocessing: readonly string[];
  /** Devices the runner can execute on. The caller must request one explicitly; none is chosen silently. */
  readonly devices: readonly Device[];
  /** Precisions the runner can execute with. `mixed` implies CUDA training. */
  readonly precisions: readonly Precision[];
  /** Trains in bounded, checkpointable chunks through trainChunk and exportModel. */
  readonly iterative: boolean;
}
// Backend methods do computation only. They never claim jobs or mutate durable state.
export interface MlBackend {
  readonly id: string;
  readonly capabilities: BackendCapabilities;
  validate(snapshot: Snapshot): void;
  prepare(snapshot: Snapshot, rows: Row[], signal: AbortSignal): Promise<PreparedData>;
  train(snapshot: Snapshot, rows: Row[], prepared: PreparedData, signal: AbortSignal): Promise<TrainedModel>;
  evaluate(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<MetricValue[]>;
  /** Optional richer form that also reports measured timings; the processor prefers it when present. */
  evaluateDetailed?(snapshot: Snapshot, rows: Row[], prepared: PreparedData, model: TrainedModel, signal: AbortSignal): Promise<Evaluation>;
  predict(snapshot: Snapshot, artifact: Buffer, rows: Row[], environment: Record<string, string>, signal: AbortSignal): Promise<unknown[]>;
  /**
   * Iterative (epoch-based) models only. Meticule owns the epoch loop: it asks for bounded chunks, persists each
   * returned checkpoint under lease fencing, and decides early stopping. The backend never persists anything itself.
   */
  trainChunk?(snapshot: Snapshot, rows: Row[], prepared: PreparedData, state: Buffer | null, epochs: number, signal: AbortSignal): Promise<ChunkResult>;
  /** Builds the inference artifact from a checkpoint, optionally from its best-epoch weights. */
  exportModel?(snapshot: Snapshot, checkpoint: Buffer, restoreBest: boolean, signal: AbortSignal): Promise<TrainedModel>;
}
/** One bounded chunk of training. `checkpoint` is opaque to the core and versioned by the backend's own format. */
export interface ChunkResult {
  checkpoint: Buffer;
  checkpointFormat: string;
  checkpointSchema: number;
  /** Epochs completed in total, including earlier chunks. */
  completedEpoch: number;
  /** Summaries of only the epochs this chunk ran. */
  epochs: { epoch: number; trainLoss: number; validationLoss: number | null; elapsedSeconds: number }[];
  environment: Record<string, string>;
}
