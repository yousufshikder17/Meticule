import { createHash } from "node:crypto";

export interface DocumentChunk { index: number; content: string; tokenEstimate: number; contentHash: string }
export const sha256 = (value: string): string => createHash("sha256").update(value, "utf8").digest("hex");

export function extractText(mediaType: "text/plain" | "text/markdown" | "application/json", content: string): string {
  if (mediaType === "application/json") {
    let parsed: unknown; try { parsed = JSON.parse(content); } catch { throw new Error("Document content is not valid JSON"); }
    return JSON.stringify(parsed, null, 2);
  }
  return content.replace(/\r\n?/g, "\n").trim();
}

export function chunkText(text: string, maximumCharacters = 1_600, overlapCharacters = 200): DocumentChunk[] {
  if (maximumCharacters < 256 || overlapCharacters < 0 || overlapCharacters >= maximumCharacters) throw new Error("Invalid chunk configuration");
  const normalized = text.replace(/[ \t]+/g, " ").replace(/\n{3,}/g, "\n\n").trim();
  if (!normalized) throw new Error("Document extraction produced no text");
  const chunks: DocumentChunk[] = []; let start = 0;
  while (start < normalized.length) {
    let end = Math.min(normalized.length, start + maximumCharacters);
    if (end < normalized.length) {
      const boundary = Math.max(normalized.lastIndexOf("\n", end), normalized.lastIndexOf(" ", end));
      if (boundary > start + Math.floor(maximumCharacters / 2)) end = boundary;
    }
    const content = normalized.slice(start, end).trim();
    if (content) chunks.push({ index: chunks.length, content, tokenEstimate: Math.max(1, Math.ceil(content.length / 4)), contentHash: sha256(content) });
    if (end >= normalized.length) break;
    start = Math.max(start + 1, end - overlapCharacters);
  }
  return chunks;
}
