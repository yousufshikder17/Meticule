import type pg from "pg";
import type { Principal } from "../db/types.js";
import { AuthorizationError, ConflictError, NotFoundError } from "../domain/errors.js";
import type { EmbeddingProvider } from "./embedding-provider.js";
import { chunkText, extractText, sha256, type DocumentChunk } from "./chunker.js";
import { DeleteDocumentSchema, IngestDocumentSchema, SearchRetrievalSchema, type RetrievalResult } from "./retrieval-schema.js";

type Row = Record<string, unknown>;
export interface DocumentRecord {
  id: string; tenantId: string; logicalId: string; version: number; ownerUserId: string; visibility: "private" | "tenant";
  title: string; mediaType: string; sourceUri: string | null; contentHash: string; metadata: Record<string, string | number | boolean | null>;
  status: "pending" | "ready" | "failed" | "deleted"; embeddingProvider: string; embeddingModel: string; embeddingDimension: number;
  supersedesId: string | null; isCurrent: boolean; errorDetails: unknown; deletedAt: Date | null; createdAt: Date; updatedAt: Date;
}
export interface RetrievalSearchOutput { results: RetrievalResult[]; totalTokens: number; embeddingProvider: string; embeddingModel: string }

export class RetrievalService {
  constructor(private readonly pool: pg.Pool, readonly provider: EmbeddingProvider) {}

  async ingest(principal: Principal, raw: unknown, signal = new AbortController().signal, supersedesId?: string): Promise<DocumentRecord> {
    const input = IngestDocumentSchema.parse(raw);
    if (input.visibility === "tenant" && !principal.roles.includes("document_manager")) throw new AuthorizationError("Tenant-visible documents require document_manager role");
    const extracted = extractText(input.mediaType, input.content); const chunks = chunkText(extracted);
    const logicalId = crypto.randomUUID();
    const pending = await this.createPending(principal, input, extracted, supersedesId, logicalId);
    try {
      const vectors: number[][] = [];
      for (let offset = 0; offset < chunks.length; offset += 64) {
        if (signal.aborted) throw signal.reason ?? new Error("Document ingestion cancelled");
        const embedded = await this.provider.embed(chunks.slice(offset, offset + 64).map((chunk) => chunk.content), signal);
        vectors.push(...embedded.vectors);
      }
      return await this.completeIngestion(principal, pending, chunks, vectors);
    } catch (error) {
      await this.failIngestion(principal, pending, error);
      throw error;
    }
  }

  async list(principal: Principal): Promise<DocumentRecord[]> {
    const result = await this.pool.query(
      `SELECT * FROM documents WHERE tenant_id=$1 AND status<>'deleted' AND (visibility='tenant' OR owner_user_id=$2)
       ORDER BY logical_id,version DESC`, [principal.tenantId, principal.userId],
    );
    return result.rows.map((row) => this.fromRow(row));
  }

  async versions(principal: Principal, documentId: string): Promise<DocumentRecord[]> {
    const selected = await this.visible(principal, documentId, true);
    const result = await this.pool.query("SELECT * FROM documents WHERE tenant_id=$1 AND logical_id=$2 ORDER BY version DESC", [principal.tenantId, selected.logicalId]);
    return result.rows.map((row) => this.fromRow(row));
  }

  async delete(principal: Principal, documentId: string, raw: unknown = {}): Promise<number> {
    const input = DeleteDocumentSchema.parse(raw); const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const selected = await client.query("SELECT * FROM documents WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, documentId]);
      if (!selected.rowCount) throw new NotFoundError("Document not found");
      this.authorizeManage(principal, selected.rows[0]);
      const documents = await client.query(
        `UPDATE documents SET status='deleted',is_current=false,deleted_at=COALESCE(deleted_at,now()),deleted_by=COALESCE(deleted_by,$1),
         deletion_reason=COALESCE(deletion_reason,$2),updated_at=now() WHERE tenant_id=$3 AND logical_id=$4 AND status<>'deleted' RETURNING id`,
        [principal.userId, input.reason, principal.tenantId, selected.rows[0].logical_id],
      );
      const ids = documents.rows.map((row) => row.id as string);
      if (ids.length) await client.query("UPDATE document_chunks SET deleted_at=COALESCE(deleted_at,now()) WHERE tenant_id=$1 AND document_id=ANY($2::uuid[])", [principal.tenantId, ids]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'document.deleted',$3)", [principal.tenantId, principal.userId, JSON.stringify({ logicalId: selected.rows[0].logical_id, versions: ids.length, reason: input.reason })]);
      await client.query("COMMIT"); return ids.length;
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  async search(principal: Principal, raw: unknown, signal = new AbortController().signal): Promise<RetrievalSearchOutput> {
    const input = SearchRetrievalSchema.parse(raw);
    const embedded = await this.provider.embed([input.query], signal); const firstVector = embedded.vectors.at(0);
    if (!firstVector) throw new Error("Embedding provider returned no query vector");
    const queryVector = this.vector(firstVector);
    const result = await this.pool.query(
      `WITH ranked AS (
         SELECT c.id AS chunk_id,c.chunk_index,c.content_text,c.token_estimate,c.content_hash,c.metadata AS chunk_metadata,
                d.id AS document_id,d.logical_id,d.version,d.title,d.source_uri,d.metadata,
                1-(c.embedding <=> $4::vector) AS score
         FROM document_chunks c JOIN documents d ON d.tenant_id=c.tenant_id AND d.id=c.document_id
         WHERE c.tenant_id=$1 AND c.deleted_at IS NULL AND d.status='ready' AND d.is_current
           AND (d.visibility='tenant' OR d.owner_user_id=$2)
           AND c.embedding_provider=$5 AND c.embedding_model=$6 AND c.embedding_dimension=$7
           AND d.metadata @> $8::jsonb
       ) SELECT * FROM ranked WHERE score >= $3 ORDER BY score DESC,document_id,chunk_index LIMIT $9`,
      [principal.tenantId, principal.userId, input.minimumScore, queryVector, this.provider.id, this.provider.model, this.provider.dimensions, JSON.stringify(input.metadata), input.maxChunks * 10],
    );
    const results: RetrievalResult[] = []; let totalTokens = 0;
    for (const row of result.rows) {
      const tokens = Number(row.token_estimate); if (results.length >= input.maxChunks || totalTokens + tokens > input.maxTokens) continue;
      results.push({ content: row.content_text, score: Number(row.score), tokenEstimate: tokens, metadata: row.metadata,
        citation: { documentId: row.document_id, logicalId: row.logical_id, version: Number(row.version), chunkId: row.chunk_id, chunkIndex: Number(row.chunk_index), title: row.title, sourceUri: row.source_uri, contentHash: row.content_hash } });
      totalTokens += tokens;
    }
    return { results, totalTokens, embeddingProvider: this.provider.id, embeddingModel: this.provider.model };
  }

  private async createPending(principal: Principal, input: ReturnType<typeof IngestDocumentSchema.parse>, extracted: string, supersedesId: string | undefined, initialLogicalId: string): Promise<DocumentRecord> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN"); let logicalId = initialLogicalId; let version = 1; let priorId: string | null = null;
      if (supersedesId) {
        const prior = await client.query("SELECT * FROM documents WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, supersedesId]);
        if (!prior.rowCount) throw new NotFoundError("Document not found"); this.authorizeManage(principal, prior.rows[0]);
        logicalId = prior.rows[0].logical_id; priorId = prior.rows[0].id;
        const pending = await client.query("SELECT 1 FROM documents WHERE tenant_id=$1 AND logical_id=$2 AND status='pending' FOR UPDATE", [principal.tenantId, logicalId]);
        if (pending.rowCount) throw new ConflictError("A document version is already being ingested");
        const maximum = await client.query("SELECT max(version) AS version FROM documents WHERE tenant_id=$1 AND logical_id=$2", [principal.tenantId, logicalId]); version = Number(maximum.rows[0].version) + 1;
      }
      const inserted = await client.query(
        `INSERT INTO documents(tenant_id,logical_id,version,owner_user_id,visibility,title,media_type,source_uri,content_text,content_hash,metadata,status,embedding_provider,embedding_model,embedding_dimension,supersedes_id,is_current)
         VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,'pending',$12,$13,$14,$15,false) RETURNING *`,
        [principal.tenantId, logicalId, version, principal.userId, input.visibility, input.title, input.mediaType, input.sourceUri, extracted, sha256(extracted), JSON.stringify(input.metadata), this.provider.id, this.provider.model, this.provider.dimensions, priorId],
      );
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'user',$2,'document.ingestion_requested',$3)", [principal.tenantId, principal.userId, JSON.stringify({ documentId: inserted.rows[0].id, logicalId, version, visibility: input.visibility, mediaType: input.mediaType })]);
      await client.query("COMMIT"); return this.fromRow(inserted.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async completeIngestion(principal: Principal, document: DocumentRecord, chunks: DocumentChunk[], vectors: number[][]): Promise<DocumentRecord> {
    if (vectors.length !== chunks.length) throw new Error("Embedding count does not match document chunks");
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      const locked = await client.query("SELECT status FROM documents WHERE tenant_id=$1 AND id=$2 FOR UPDATE", [principal.tenantId, document.id]);
      if (!locked.rowCount || locked.rows[0].status !== "pending") throw new ConflictError("Document ingestion is no longer pending");
      for (let index = 0; index < chunks.length; index += 1) {
        const chunk = chunks[index]; const vector = vectors[index];
        if (!chunk || !vector) throw new Error("Document chunk and embedding counts diverged");
        await client.query(
          `INSERT INTO document_chunks(tenant_id,document_id,chunk_index,content_text,token_estimate,content_hash,metadata,embedding,embedding_provider,embedding_model,embedding_dimension)
           VALUES($1,$2,$3,$4,$5,$6,$7,$8::vector,$9,$10,$11)`,
          [principal.tenantId, document.id, chunk.index, chunk.content, chunk.tokenEstimate, chunk.contentHash, JSON.stringify({ documentVersion: document.version }), this.vector(vector), this.provider.id, this.provider.model, this.provider.dimensions],
        );
      }
      await client.query("UPDATE documents SET is_current=false,updated_at=now() WHERE tenant_id=$1 AND logical_id=$2 AND id<>$3 AND is_current", [principal.tenantId, document.logicalId, document.id]);
      const updated = await client.query("UPDATE documents SET status='ready',is_current=true,error_details=NULL,updated_at=now() WHERE tenant_id=$1 AND id=$2 RETURNING *", [principal.tenantId, document.id]);
      await client.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'system','retrieval-service','document.ingestion_completed',$2)", [principal.tenantId, JSON.stringify({ documentId: document.id, logicalId: document.logicalId, version: document.version, chunks: chunks.length, embeddingProvider: this.provider.id, embeddingModel: this.provider.model, dimension: this.provider.dimensions })]);
      await client.query("COMMIT"); return this.fromRow(updated.rows[0]);
    } catch (error) { await client.query("ROLLBACK").catch(() => undefined); throw error; }
    finally { client.release(); }
  }

  private async failIngestion(principal: Principal, document: DocumentRecord, error: unknown): Promise<void> {
    await this.pool.query("UPDATE documents SET status='failed',is_current=false,error_details=$1,updated_at=now() WHERE tenant_id=$2 AND id=$3 AND status='pending'", [JSON.stringify({ code: "embedding_failed", message: error instanceof Error ? error.message : String(error) }), principal.tenantId, document.id]);
    await this.pool.query("INSERT INTO audit_events(tenant_id,run_id,actor_type,actor_id,event_type,details) VALUES($1,NULL,'system','retrieval-service','document.ingestion_failed',$2)", [principal.tenantId, JSON.stringify({ documentId: document.id, logicalId: document.logicalId, version: document.version, code: "embedding_failed" })]);
  }

  private async visible(principal: Principal, id: string, includeDeleted = false): Promise<DocumentRecord> {
    const result = await this.pool.query(`SELECT * FROM documents WHERE tenant_id=$1 AND id=$2 ${includeDeleted ? "" : "AND status<>'deleted'"}`, [principal.tenantId, id]);
    if (!result.rowCount) throw new NotFoundError("Document not found");
    const document = this.fromRow(result.rows[0]);
    if (document.visibility === "private" && document.ownerUserId !== principal.userId) throw new NotFoundError("Document not found");
    return document;
  }
  private authorizeManage(principal: Principal, row: Row): void {
    const authorized = row.visibility === "tenant" ? principal.roles.includes("document_manager") : row.owner_user_id === principal.userId;
    if (!authorized) throw new AuthorizationError("Document management is not authorized");
  }
  private vector(values: number[]): string { if (values.length !== this.provider.dimensions || values.some((value) => !Number.isFinite(value))) throw new Error("Embedding vector has invalid dimensions or values"); return `[${values.join(",")}]`; }
  private fromRow(row: Row): DocumentRecord { return { id: row.id as string, tenantId: row.tenant_id as string, logicalId: row.logical_id as string, version: Number(row.version), ownerUserId: row.owner_user_id as string, visibility: row.visibility as DocumentRecord["visibility"], title: row.title as string, mediaType: row.media_type as string, sourceUri: row.source_uri as string | null, contentHash: row.content_hash as string, metadata: row.metadata as DocumentRecord["metadata"], status: row.status as DocumentRecord["status"], embeddingProvider: row.embedding_provider as string, embeddingModel: row.embedding_model as string, embeddingDimension: Number(row.embedding_dimension), supersedesId: row.supersedes_id as string | null, isCurrent: row.is_current as boolean, errorDetails: row.error_details, deletedAt: row.deleted_at as Date | null, createdAt: row.created_at as Date, updatedAt: row.updated_at as Date }; }
}
