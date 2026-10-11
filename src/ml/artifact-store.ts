import { createHash } from "node:crypto";
import type pg from "pg";
import { z } from "zod";
import { NotFoundError } from "../domain/errors.js";
import type { ArtifactStore } from "./backend.js";
import { MlBackendFailure } from "./contract.js";
import { ArtifactReferenceSchema, type ArtifactReference } from "./domain.js";

/** Public storage bound for one blob: datasets, model artifacts and training checkpoints alike. */
export const MAX_ARTIFACT_BYTES = 16_000_000;

// ponytail: PostgreSQL blobs capped at 16 MB; use an object-store adapter for larger artifacts.
export class PostgresArtifactStore implements ArtifactStore {
  constructor(private readonly database: pg.Pool) {}
  async put(tenantId: string, bytes: Uint8Array, mediaType: string): Promise<ArtifactReference> {
    z.uuid().parse(tenantId);
    // Classified, so the limit reaches a stored training failure as a clear reason rather than a generic message.
    if (!bytes.byteLength) throw new MlBackendFailure("invalid_response", "ML artifact must not be empty");
    if (bytes.byteLength > MAX_ARTIFACT_BYTES) throw new MlBackendFailure("output_limit_exceeded", `ML artifact of ${bytes.byteLength} bytes exceeds the 16 MB public storage limit`);
    const sha256 = createHash("sha256").update(bytes).digest("hex");
    const reference = ArtifactReferenceSchema.parse({ key: sha256, sha256, size: bytes.byteLength, mediaType });
    await this.database.query("INSERT INTO ml_blobs(tenant_id,content_hash,media_type,content) VALUES($1,$2,$3,$4) ON CONFLICT DO NOTHING", [tenantId, sha256, mediaType, Buffer.from(bytes)]);
    return reference;
  }
  async get(tenantId: string, raw: ArtifactReference): Promise<Buffer> {
    z.uuid().parse(tenantId); const reference = ArtifactReferenceSchema.parse(raw);
    if (reference.key !== reference.sha256) throw new Error("Artifact key/hash mismatch");
    const result = await this.database.query("SELECT content FROM ml_blobs WHERE tenant_id=$1 AND content_hash=$2", [tenantId, reference.key]);
    if (!result.rowCount) throw new NotFoundError("ML artifact not found");
    const content = result.rows[0].content as Buffer;
    if (content.length !== reference.size || createHash("sha256").update(content).digest("hex") !== reference.sha256) throw new Error("Artifact checksum mismatch");
    return content;
  }
}
