import { createHash } from "node:crypto";
import type pg from "pg";
import { expect, it } from "vitest";
import { PostgresArtifactStore } from "../../src/ml/artifact-store.js";

it("validates references and rejects corrupted bytes or size before model loading", async () => {
  const tenant = "11111111-1111-4111-8111-111111111111";
  const bytes = Buffer.from("model"), sha256 = createHash("sha256").update(bytes).digest("hex");
  const reference = { key: sha256, sha256, size: bytes.length, mediaType: "application/octet-stream" };
  let content = bytes;
  const database = { query: async () => ({ rowCount: 1, rows: [{ content }] }) } as unknown as pg.Pool;
  const store = new PostgresArtifactStore(database);
  expect(await store.get(tenant, reference)).toEqual(bytes);
  await expect(store.get(tenant, { ...reference, key: "../escape" })).rejects.toThrow();
  await expect(store.get(tenant, { ...reference, size: 999 })).rejects.toThrow("checksum");
  content = Buffer.from("other"); await expect(store.get(tenant, reference)).rejects.toThrow("checksum");
  await expect(store.put(tenant, Buffer.alloc(0), reference.mediaType)).rejects.toThrow();
  await expect(store.put(tenant, Buffer.alloc(16_000_001), reference.mediaType)).rejects.toThrow();
});
