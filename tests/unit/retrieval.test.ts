import { afterEach, describe, expect, it } from "vitest";
import { chunkText, extractText } from "../../src/retrieval/chunker.js";
import { createRetrievalService } from "../../src/retrieval/retrieval-configuration.js";
import { OllamaEmbeddingProvider } from "../../src/retrieval/ollama-embedding-provider.js";

const originalFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = originalFetch; });

describe("retrieval components", () => {
  it("extracts JSON and chunks deterministically with bounded overlap", () => {
    expect(extractText("application/json", "{\"answer\":42}")).toContain('"answer": 42');
    expect(() => extractText("application/json", "not-json")).toThrow(/valid JSON/);
    const text = `alpha ${"evidence ".repeat(400)}`; const first = chunkText(text, 300, 40); const second = chunkText(text, 300, 40);
    expect(first).toEqual(second); expect(first.length).toBeGreaterThan(2); expect(first.every((chunk) => chunk.content.length <= 300 && chunk.tokenEstimate > 0)).toBe(true);
  });

  it("requires explicit real embedding configuration and never installs a fallback", () => {
    expect(createRetrievalService({} as never, {})).toBeNull();
    expect(() => createRetrievalService({} as never, { OLLAMA_EMBEDDING_MODEL: "embedding-model" })).toThrow(/configuration/);
    const configured = createRetrievalService({} as never, { OLLAMA_EMBEDDING_MODEL: "embedding-model", OLLAMA_EMBEDDING_DIMENSIONS: "3", OLLAMA_EMBEDDING_BASE_URL: "http://localhost:11434" });
    expect(configured?.provider).toMatchObject({ id: "ollama-embeddings", model: "embedding-model", dimensions: 3 });
  });

  it("validates Ollama embedding count and dimensions", async () => {
    globalThis.fetch = async () => new Response(JSON.stringify({ model: "embed", embeddings: [[0.1, 0.2, 0.3]], prompt_eval_count: 4 }), { status: 200, headers: { "content-type": "application/json" } });
    const provider = new OllamaEmbeddingProvider({ baseUrl: "http://localhost:11434", model: "embed", dimensions: 3, timeoutMs: 1000 });
    expect(await provider.embed(["text"], new AbortController().signal)).toMatchObject({ vectors: [[0.1, 0.2, 0.3]], inputTokens: 4 });
    globalThis.fetch = async () => new Response(JSON.stringify({ embeddings: [[0.1, 0.2]] }), { status: 200 });
    await expect(provider.embed(["text"], new AbortController().signal)).rejects.toThrow(/dimension/);
  });

  it("does not surface provider response bodies in normalized errors", async () => {
    globalThis.fetch = async () => new Response("sensitive upstream detail", { status: 500 });
    const provider = new OllamaEmbeddingProvider({ baseUrl: "http://localhost:11434", model: "embed", dimensions: 3, timeoutMs: 1000 });
    const error = await provider.embed(["text"], new AbortController().signal).catch((value: unknown) => value as Error);
    expect(error.message).toContain("HTTP 500"); expect(error.message).not.toContain("sensitive upstream detail");
  });

  it("fails before transport when embedding cancellation is already requested", async () => {
    let called = false; globalThis.fetch = async () => { called = true; return new Response(); };
    const provider = new OllamaEmbeddingProvider({ baseUrl: "http://localhost:11434", model: "embed", dimensions: 3, timeoutMs: 1000 });
    const controller = new AbortController(); controller.abort(new Error("synthetic cancellation"));
    await expect(provider.embed(["text"], controller.signal)).rejects.toMatchObject({ code: "cancelled", retryable: false });
    expect(called).toBe(false);
  });
});
