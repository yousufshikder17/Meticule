import { z } from "zod";
import type pg from "pg";
import type { Principal } from "../db/types.js";
import { ConflictError, NotFoundError } from "../domain/errors.js";
import type { ArtifactStore, MlBackend } from "./backend.js";
import { MlBackendFailure } from "./contract.js";
import { DatasetVersion, TrainingSnapshot, validateRows, type Row } from "./domain.js";
import { getModel } from "./model-catalog.js";
import { audit, authorizeMl, transaction } from "./persistence.js";
import { HIGHER_IS_CLOSER, SIMILARITY_METRICS, similarityMatrix, type SimilarityMetric } from "./similarity.js";
import { embeddingDimension } from "./tasks.js";

/**
 * Embedding indexes and similarity (Phase 4 M6). Generation stays in the backends; this service owns persistence and search.
 * An index holds the vectors of one dataset version embedded by one pinned READY model version, in PostgreSQL with pgvector:
 * built in bounded steps (one runner call of at most BUILD_BATCH rows per request), then READY and append-only, and searched
 * exactly (no approximate index). Every query filters tenant_id; another tenant's index or endpoint is simply not found.
 */
export const MAX_INDEX_ROWS = 5000, BUILD_BATCH = 256, MAX_QUERY_ROWS = 16, MAX_K = 100, MAX_PAIR_ROWS = 64;
// Fixed operator per stored metric (never interpolated from input): cosine distance, negative inner product, euclidean distance.
const OPERATORS: Record<SimilarityMetric, { operator: string; score: (distance: number) => number }> = {
  cosine: { operator: "<=>", score: d => 1 - d }, dot: { operator: "<#>", score: d => -d }, euclidean: { operator: "<->", score: d => d },
};
const PAYLOAD_TYPES = new Set(["number", "string", "boolean", "text", "timestamp"]);

export const CreateEmbeddingIndex = z.object({
  name: z.string().trim().min(1).max(120), endpointId: z.uuid(), datasetVersionId: z.uuid(), metric: z.enum(SIMILARITY_METRICS).default("cosine"),
  payloadColumns: z.array(z.string().min(1).max(64)).max(16).default([]),
});
export const SearchEmbeddingIndex = z.object({ rows: z.array(z.unknown()).min(1).max(MAX_QUERY_ROWS), k: z.int().min(1).max(MAX_K).default(10) });
export const PairwiseSimilarity = z.object({ left: z.array(z.unknown()).min(1).max(MAX_PAIR_ROWS), right: z.array(z.unknown()).min(1).max(MAX_PAIR_ROWS), metric: z.enum(SIMILARITY_METRICS).default("cosine") });

interface PinnedModel { id: string; status: string; snapshot: unknown; environment: Record<string, string>; content: unknown; format: string }

/** A vector the index may hold: the index's dimension, finite, and non-zero where cosine would divide by its norm. */
function checkVector(vector: unknown, dimension: number, metric: SimilarityMetric): vector is number[] {
  if (!Array.isArray(vector) || vector.length !== dimension || !vector.every(x => typeof x === "number" && Number.isFinite(x))) return false;
  return metric !== "cosine" || vector.some(x => x !== 0);
}
const literal = (vector: readonly number[]) => `[${vector.join(",")}]`;

export class MlEmbeddingService {
  constructor(private readonly pool: pg.Pool, private readonly artifacts: ArtifactStore, private readonly backends: Map<string, MlBackend>) {}

  async create(principal: Principal, raw: unknown) {
    authorizeMl(principal, true); const input = CreateEmbeddingIndex.parse(raw);
    return transaction(this.pool, async c => {
      const endpoint = await c.query(`SELECT e.id, m.id AS model_version_id, m.status, t.snapshot, t.environment FROM ml_endpoints e
        JOIN ml_model_versions m ON m.tenant_id=e.tenant_id AND m.id=e.model_version_id JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id
        WHERE e.tenant_id=$1 AND e.id=$2 AND e.status='ACTIVE'`, [principal.tenantId, input.endpointId]);
      if (!endpoint.rowCount) throw new NotFoundError("Active endpoint not found");
      const model = endpoint.rows[0], snapshot = TrainingSnapshot.parse(model.snapshot);
      if (model.status !== "READY") throw new ConflictError("Embedding indexes need a READY model version");
      if (getModel(snapshot.backend, snapshot.algorithm).taskType !== "embedding") throw new ConflictError("The endpoint's model does not produce embeddings");
      const dimension = embeddingDimension(model.environment);
      const versionRow = await c.query("SELECT document FROM ml_dataset_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, input.datasetVersionId]);
      if (!versionRow.rowCount) throw new NotFoundError("Dataset version not found");
      const dataset = DatasetVersion.parse(versionRow.rows[0].document);
      if (dataset.rowCount > MAX_INDEX_ROWS) throw new ConflictError(`Embedding indexes hold at most ${MAX_INDEX_ROWS} rows`);
      const columns = new Map(dataset.schema.columns.map(col => [col.name, col.type]));
      for (const name of snapshot.pipeline.definition.features) {
        const type = snapshot.dataset.schema.columns.find(col => col.name === name)!.type;
        if (columns.get(name) !== type) throw new ConflictError("The dataset lacks the model's feature columns with the same types");
      }
      for (const name of input.payloadColumns) if (!PAYLOAD_TYPES.has(columns.get(name) ?? "")) throw new ConflictError("Payload columns must be existing non-media dataset columns");
      if (new Set(input.payloadColumns).size !== input.payloadColumns.length) throw new ConflictError("Payload columns must be distinct");
      const duplicate = await c.query("SELECT 1 FROM ml_embedding_indexes WHERE tenant_id=$1 AND name=$2", [principal.tenantId, input.name]);
      if (duplicate.rowCount) throw new ConflictError("An embedding index with this name already exists");
      const index = await c.query(`INSERT INTO ml_embedding_indexes(tenant_id,name,endpoint_id,model_version_id,dataset_version_id,dimension,metric,row_count,payload_columns,created_by)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10) RETURNING *`, [principal.tenantId, input.name, input.endpointId, model.model_version_id, input.datasetVersionId, dimension, input.metric, dataset.rowCount, JSON.stringify(input.payloadColumns), principal.userId]);
      await audit(c, principal.tenantId, principal.userId, "ml.embedding_index_created", { indexId: index.rows[0].id, endpointId: input.endpointId, rows: dataset.rowCount, dimension, metric: input.metric });
      return { ...index.rows[0], built: 0 };
    });
  }

  async get(principal: Principal, indexId: string) {
    authorizeMl(principal);
    const index = await this.pool.query("SELECT i.*, (SELECT count(*)::int FROM ml_embedding_vectors v WHERE v.tenant_id=i.tenant_id AND v.index_id=i.id) AS built FROM ml_embedding_indexes i WHERE i.tenant_id=$1 AND i.id=$2", [principal.tenantId, indexId]);
    if (!index.rowCount) throw new NotFoundError("Embedding index not found");
    return index.rows[0];
  }

  /**
   * Embeds the next missing batch (at most BUILD_BATCH rows, in dataset row order) in one runner call while holding the index
   * row (FOR UPDATE NOWAIT: a concurrent build gets a conflict at once), appends the vectors and marks the index READY when it
   * is complete. A runner that returns an unusable vector marks the index FAILED, and that outcome is committed.
   */
  async build(principal: Principal, indexId: string, signal: AbortSignal) {
    authorizeMl(principal, true);
    return transaction(this.pool, async c => {
      let index;
      try { index = (await c.query("SELECT * FROM ml_embedding_indexes WHERE tenant_id=$1 AND id=$2 FOR UPDATE NOWAIT", [principal.tenantId, indexId])).rows[0]; }
      catch (error) { if ((error as { code?: string }).code === "55P03") throw new ConflictError("Embedding index is being built by another request"); throw error; }
      if (!index) throw new NotFoundError("Embedding index not found");
      if (index.status !== "BUILDING") throw new ConflictError(`Embedding index is ${index.status}`);
      const built = (await c.query("SELECT count(*)::int AS n FROM ml_embedding_vectors WHERE tenant_id=$1 AND index_id=$2", [principal.tenantId, indexId])).rows[0].n as number;
      const model = await this.pinned(c, principal.tenantId, index.model_version_id);
      const version = DatasetVersion.parse((await c.query("SELECT document FROM ml_dataset_versions WHERE tenant_id=$1 AND id=$2", [principal.tenantId, index.dataset_version_id])).rows[0].document);
      const rows = JSON.parse((await this.artifacts.get(principal.tenantId, version.content)).toString("utf8")) as Row[];
      if (rows.length !== index.row_count) throw new MlBackendFailure("invalid_response", "Dataset row count differs from the index");
      const batch = rows.slice(built, built + BUILD_BATCH);
      const snapshot = TrainingSnapshot.parse(model.snapshot), features = snapshot.pipeline.definition.features;
      const vectors = await this.embed(principal.tenantId, model, batch.map(row => Object.fromEntries(features.map(f => [f, row[f] ?? null]))), signal);
      if (vectors.length !== batch.length || !vectors.every(v => checkVector(v, index.dimension, index.metric))) {
        await c.query("UPDATE ml_embedding_indexes SET status='FAILED' WHERE tenant_id=$1 AND id=$2", [principal.tenantId, indexId]);
        await audit(c, principal.tenantId, principal.userId, "ml.embedding_index_failed", { indexId, reason: "invalid_response" });
        return { ...index, status: "FAILED", built, failure: { category: "invalid_response", message: "The model returned vectors the index cannot hold" } };
      }
      const payloadColumns = index.payload_columns as string[];
      for (let i = 0; i < batch.length; i++) {
        const payload = Object.fromEntries(payloadColumns.map(col => [col, batch[i]![col] ?? null]));
        await c.query("INSERT INTO ml_embedding_vectors(tenant_id,index_id,dimension,row_index,embedding,payload) VALUES($1,$2,$3,$4,$5::vector,$6)",
          [principal.tenantId, indexId, index.dimension, built + i, literal(vectors[i]!), JSON.stringify(payload)]);
      }
      const total = built + batch.length, ready = total === index.row_count;
      const updated = ready ? (await c.query("UPDATE ml_embedding_indexes SET status='READY', ready_at=now() WHERE tenant_id=$1 AND id=$2 RETURNING *", [principal.tenantId, indexId])).rows[0] : index;
      await audit(c, principal.tenantId, principal.userId, "ml.embedding_index_built", { indexId, rows: batch.length, built: total, ready });
      return { ...updated, built: total };
    });
  }

  /** Exact nearest rows of a READY index for each query row, embedded with the index's pinned model version. */
  async search(principal: Principal, indexId: string, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal); const input = SearchEmbeddingIndex.parse(raw);
    const index = (await this.pool.query("SELECT * FROM ml_embedding_indexes WHERE tenant_id=$1 AND id=$2", [principal.tenantId, indexId])).rows[0];
    if (!index) throw new NotFoundError("Embedding index not found");
    if (index.status !== "READY") throw new ConflictError("Only READY embedding indexes can be searched");
    const model = await this.pinned(this.pool, principal.tenantId, index.model_version_id);
    const snapshot = TrainingSnapshot.parse(model.snapshot);
    const vectors = await this.embed(principal.tenantId, model, validateRows(snapshot.dataset.schema, input.rows, true), signal);
    const metric = index.metric as SimilarityMetric, { operator, score } = OPERATORS[metric];
    const results = [];
    for (const vector of vectors) {
      if (!Array.isArray(vector) || vector.length !== index.dimension) throw new ConflictError("Query vector dimension differs from the index");
      if (!checkVector(vector, index.dimension, metric)) throw new ConflictError("Query vector is not usable with this metric");
      const rows = await this.pool.query(`SELECT row_index, payload, (embedding ${operator} $3::vector) AS distance FROM ml_embedding_vectors WHERE tenant_id=$1 AND index_id=$2 ORDER BY distance, row_index LIMIT $4`,
        [principal.tenantId, indexId, literal(vector), input.k]);
      results.push(rows.rows.map(r => ({ rowIndex: r.row_index as number, score: score(Number(r.distance)), payload: r.payload })));
    }
    await transaction(this.pool, c => audit(c, principal.tenantId, principal.userId, "ml.embedding_index_searched", { indexId, queries: vectors.length, k: input.k }));
    return { metric, closer: HIGHER_IS_CLOSER[metric] ? "higher" : "lower", results };
  }

  /** left x right similarity of rows embedded through an ACTIVE endpoint whose READY model produces embeddings. */
  async pairwise(principal: Principal, endpointId: string, raw: unknown, signal: AbortSignal) {
    authorizeMl(principal); const input = PairwiseSimilarity.parse(raw);
    const endpoint = (await this.pool.query("SELECT model_version_id FROM ml_endpoints WHERE tenant_id=$1 AND id=$2 AND status='ACTIVE'", [principal.tenantId, endpointId])).rows[0];
    if (!endpoint) throw new NotFoundError("Active endpoint not found");
    const model = await this.pinned(this.pool, principal.tenantId, endpoint.model_version_id);
    const snapshot = TrainingSnapshot.parse(model.snapshot);
    const left = await this.embed(principal.tenantId, model, validateRows(snapshot.dataset.schema, input.left, true), signal);
    const right = await this.embed(principal.tenantId, model, validateRows(snapshot.dataset.schema, input.right, true), signal);
    const matrix = similarityMatrix(input.metric, left, right);
    await transaction(this.pool, c => audit(c, principal.tenantId, principal.userId, "ml.embedding_similarity", { endpointId, left: left.length, right: right.length, metric: input.metric }));
    return { metric: input.metric, closer: HIGHER_IS_CLOSER[input.metric] ? "higher" : "lower", matrix };
  }

  private async pinned(c: Pick<pg.Pool, "query">, tenant: string, modelVersionId: string): Promise<PinnedModel> {
    const result = await c.query(`SELECT m.id, m.status, t.snapshot, t.environment, a.content, a.format FROM ml_model_versions m
      JOIN ml_training_runs t ON t.tenant_id=m.tenant_id AND t.id=m.training_run_id JOIN ml_artifacts a ON a.tenant_id=m.tenant_id AND a.id=m.artifact_id
      WHERE m.tenant_id=$1 AND m.id=$2`, [tenant, modelVersionId]);
    const model = result.rows[0] as PinnedModel | undefined;
    if (!model) throw new NotFoundError("Model version not found");
    if (model.status !== "READY") throw new ConflictError("The embedding model version is no longer READY");
    const snapshot = TrainingSnapshot.parse(model.snapshot);
    if (getModel(snapshot.backend, snapshot.algorithm).taskType !== "embedding") throw new ConflictError("The model does not produce embeddings");
    return model;
  }

  /** One runner call; the backend validates the vectors against the recorded embedding dimension (E1). */
  private async embed(tenant: string, model: PinnedModel, rows: Row[], signal: AbortSignal): Promise<number[][]> {
    const snapshot = TrainingSnapshot.parse(model.snapshot);
    const backend = this.backends.get(snapshot.backend);
    if (!backend) throw new ConflictError("Embedding backend is not configured");
    if (!backend.capabilities.artifactFormats.includes(model.format)) throw new MlBackendFailure("incompatible_artifact", "Unsupported artifact format");
    const bytes = await this.artifacts.get(tenant, model.content as Parameters<ArtifactStore["get"]>[1]);
    return (await backend.predict(snapshot, bytes, rows, model.environment, signal)) as number[][];
  }
}
