import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { isDeepStrictEqual } from "node:util";
import { describe, expect, it } from "vitest";
import { ConflictError } from "../../src/domain/errors.js";
import { DatasetSchema, Rows, validateRows } from "../../src/ml/domain.js";
import { MEDIA_BUDGETS, MEDIA_LIMITS, MediaBudget, decodeBase64, imageHeader, timestampMs, wavLayout, type WavLayout } from "../../src/ml/media.js";

type Kind = "base64" | "image" | "audio" | "timestamp";
type Vector = { id: string; kind: Kind; input: string | { parts: [string, number][] }; verdict: "accept" | "refuse"; note: string; facts?: Record<string, unknown> };
// Shared with ml/media_common.py and parse_timestamp (tests/python/check_media.py): both halves decide every vector alike.
const document = JSON.parse(readFileSync(resolve("tests/fixtures/media-vectors.json"), "utf8")) as { limits: Record<string, number>; vectors: Vector[] };
const expand = (input: Vector["input"]) => typeof input === "string" ? input : input.parts.map(([text, count]) => text.repeat(count)).join("");
const ATTACKS = ["png-second-ihdr", "jpeg-lying-second-sof", "wav-second-data-chunk", "wav-lying-block-align", "wav-float-nan-sample", "wav-block-align-zero", "wav-10ms-clip",
  "b64-url-safe", "b64-whitespace", "b64-mid-string-padding", "ts-feb-30", "ts-hour-24", "ts-year-0000", "ts-offset-99-99"];
const MESSAGES = new Set(["Media cell is not valid base64", "Media cell exceeds 2 MiB", "Malformed PNG", "Malformed JPEG", "Image must be PNG or JPEG", "Image dimensions out of bounds",
  "Malformed WAV", "Unsupported WAV format", "Audio duration out of bounds", "Audio samples must be finite"]);

function factsOf(kind: Kind, text: string): Record<string, unknown> {
  if (kind === "base64") { const bytes = decodeBase64(text); return { length: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") }; }
  if (kind === "image") return { ...imageHeader(decodeBase64(text)) };
  if (kind === "audio") { const { rate, channels, frames } = wavLayout(decodeBase64(text)); return { sampleRate: rate, channels, frames }; }
  const epochMs = timestampMs(text);
  if (epochMs === undefined) throw new ConflictError("Invalid timestamp");
  return { epochMs };
}
/** Why a vector disagrees with the file, or null. Anything but a classified refusal with a fixed message is a defect. */
function disagreement(vector: Vector): string | null {
  let facts: Record<string, unknown>;
  try { facts = factsOf(vector.kind, expand(vector.input)); }
  catch (error) {
    if (!(error instanceof ConflictError) || (vector.kind !== "timestamp" && !MESSAGES.has(error.message))) return `${vector.id}: ${String(error)}`;
    return vector.verdict === "refuse" ? null : `${vector.id}: refused (${error.message})`;
  }
  if (vector.verdict === "refuse") return `${vector.id}: accepted`;
  const agrees = vector.kind === "timestamp" ? Math.abs((facts.epochMs as number) - (vector.facts!.epochMs as number)) <= 0.002 : isDeepStrictEqual(facts, vector.facts);
  return agrees ? null : `${vector.id}: ${JSON.stringify(facts)}`;
}

describe("shared media and timestamp vectors", () => {
  it("uses the vector file's limits and keeps every review attack case as a refusal", () => {
    expect(MEDIA_LIMITS).toEqual(document.limits);
    const byId = new Map(document.vectors.map(v => [v.id, v]));
    expect(byId.size).toBe(document.vectors.length);
    for (const id of ATTACKS) expect(byId.get(id)?.verdict, id).toBe("refuse");
  });

  it.each(["base64", "image", "audio", "timestamp"] as const)("decides every %s vector as the file says", kind => {
    const chosen = document.vectors.filter(v => v.kind === kind);
    expect(chosen.length).toBeGreaterThanOrEqual(10);
    expect(new Set(chosen.map(v => v.verdict))).toEqual(new Set(["accept", "refuse"]));
    expect(chosen.map(disagreement).filter(Boolean)).toEqual([]);
  });
});

/** A PNG of the given size: signature, IHDR, IEND (header walks never read pixels or CRCs). */
function png(width: number, height: number): string {
  const b = Buffer.alloc(45);
  Buffer.from("89504e470d0a1a0a0000000d49484452", "hex").copy(b);
  b.writeUInt32BE(width, 16); b.writeUInt32BE(height, 20); b[24] = 8;
  b.write("IEND", 37, "latin1");
  return b.toString("base64");
}
/** 8-bit mono PCM silence at 8000 Hz. */
function wav(seconds: number): string {
  const frames = seconds * 8000, b = Buffer.alloc(44 + frames, 128);
  b.write("RIFFxxxxWAVEfmt ", 0, "latin1"); b.writeUInt32LE(36 + frames, 4);
  b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20); b.writeUInt16LE(1, 22); b.writeUInt32LE(8000, 24); b.writeUInt32LE(8000, 28); b.writeUInt16LE(1, 32); b.writeUInt16LE(8, 34);
  b.write("data", 36, "latin1"); b.writeUInt32LE(frames, 40);
  return b.toString("base64");
}

describe("TypeScript-only media rules", () => {
  it("bounds a Rows string by the base64 length of 2 MiB", () => {
    expect(Rows.safeParse([{ photo: "A".repeat(MEDIA_LIMITS.maxBase64Chars) }]).success).toBe(true);
    expect(Rows.safeParse([{ photo: "A".repeat(MEDIA_LIMITS.maxBase64Chars + 1) }]).success).toBe(false);
  });

  it("charges images to the request's pixel budget, larger for ingest than for prediction (C5)", () => {
    expect(imageHeader(Buffer.from(png(4096, 4096), "base64"))).toEqual({ format: "png", width: 4096, height: 4096 });
    const schema = DatasetSchema.parse({ id: "11111111-1111-4111-8111-111111111111", columns: [{ name: "photo", type: "image" }, { name: "label", type: "string" }], target: "label" });
    const rows = (count: number) => Array.from({ length: count }, () => ({ photo: png(4096, 4096), label: "a" }));
    expect(validateRows(schema, rows(16))).toHaveLength(16); // 268,435,456 pixels
    expect(() => validateRows(schema, rows(17))).toThrow("Row 16: invalid photo: Images in one request exceed 268435456 pixels");
    const features = (count: number) => rows(count).map(({ photo }) => ({ photo }));
    expect(validateRows(schema, features(4), true)).toHaveLength(4); // 67,108,864 pixels
    expect(() => validateRows(schema, features(5), true)).toThrow("Row 4: invalid photo: Images in one request exceed 67108864 pixels");
  });

  it("charges audio to the request's duration budget with exact arithmetic (C5)", () => {
    const schema = DatasetSchema.parse({ id: "11111111-1111-4111-8111-111111111111", columns: [{ name: "clip", type: "audio" }] });
    const clip = wav(30);
    expect(validateRows(schema, Array.from({ length: 4 }, () => ({ clip })), true)).toHaveLength(4);
    expect(() => validateRows(schema, Array.from({ length: 5 }, () => ({ clip })), true)).toThrow("Row 4: invalid clip: Audio in one request exceeds 120 seconds");
    expect(validateRows(schema, Array.from({ length: 20 }, () => ({ clip })))).toHaveLength(20);
    expect(() => validateRows(schema, Array.from({ length: 21 }, () => ({ clip })))).toThrow("Audio in one request exceeds 600 seconds");
    // 6,000 clips of 0.1 s are exactly 600 s; a float running sum would drift past the limit.
    const tenth: WavLayout = { format: "pcm", bits: 8, channels: 1, rate: 8000, frames: 800, offset: 44 }, budget = new MediaBudget(MEDIA_BUDGETS.ingest);
    for (let i = 0; i < 6000; i++) budget.audio(tenth);
    expect(() => budget.audio({ ...tenth, frames: 1 })).toThrow(ConflictError);
  });

  it("orders timestamps a microsecond apart, even in the last allowed year", () => {
    const [a, b] = [timestampMs("2199-12-31T23:59:59.999998Z")!, timestampMs("2199-12-31T23:59:59.999999Z")!];
    expect(b).toBeGreaterThan(a);
    expect(timestampMs("2024-03-01T12:34:56Z")).toBe(Date.UTC(2024, 2, 1, 12, 34, 56));
    expect(timestampMs("2024-03-01T12:34:56.5-02:30")).toBe(Date.UTC(2024, 2, 1, 15, 4, 56, 500));
  });
});
