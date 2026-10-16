import { deflateSync } from "node:zlib";
import type { Row } from "../../src/ml/domain.js";

// Small deterministic rows per modality, shared by the Phase 4 conformance and integration tests. No hooks, no I/O.
const positive = ["good", "great", "happy", "love", "excellent"], negative = ["bad", "awful", "sad", "hate", "terrible"];
export const TEXT_ROWS: Row[] = Array.from({ length: 40 }, (_, i) => ({ x: [...(i % 2 ? positive : negative).slice(i % 3, i % 3 + 2), "the", "movie"].join(" "), label: i % 2 ? "positive" : "negative" }));

/** A solid-colour 16x16 RGB PNG, base64, built from bytes (no image library on the TypeScript side). */
export function png(r: number, g: number, b: number): string {
  const crcTable = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc = (data: Buffer) => { let c = 0xffffffff; for (const byte of data) c = crcTable[(c ^ byte) & 0xff]! ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (kind: string, data: Buffer) => { const body = Buffer.concat([Buffer.from(kind, "ascii"), data]); const out = Buffer.alloc(8 + data.length + 4); out.writeUInt32BE(data.length, 0); body.copy(out, 4); out.writeUInt32BE(crc(body), 8 + data.length); return out; };
  const size = 16, ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(size, 0); ihdr.writeUInt32BE(size, 4); ihdr.set([8, 2, 0, 0, 0], 8);
  const raw = Buffer.concat(Array.from({ length: size }, () => Buffer.concat([Buffer.from([0]), Buffer.from(Array.from({ length: size }, () => [r, g, b]).flat())])));
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]).toString("base64");
}
export const IMAGE_ROWS: Row[] = Array.from({ length: 30 }, (_, i) => ({ x: i % 2 ? png(200 + (i % 50), 20, 20) : png(20, 20, 200 + (i % 50)), label: i % 2 ? "red" : "blue" }));


/** A mono PCM16 WAV clip (base64) of a sine tone; seconds >= 0.1 so it passes the clip-length rule. */
export function wav(frequency: number, seconds = 0.3, rate = 16000, amplitude = 0.5): string {
  const frames = Math.round(rate * seconds), data = Buffer.alloc(frames * 2);
  for (let i = 0; i < frames; i++) data.writeInt16LE(Math.round(amplitude * 32767 * Math.sin((2 * Math.PI * frequency * i) / rate)), i * 2);
  const header = Buffer.alloc(44);
  header.write("RIFF", 0, "ascii"); header.writeUInt32LE(36 + data.length, 4); header.write("WAVE", 8, "ascii"); header.write("fmt ", 12, "ascii");
  header.writeUInt32LE(16, 16); header.writeUInt16LE(1, 20); header.writeUInt16LE(1, 22); header.writeUInt32LE(rate, 24); header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32); header.writeUInt16LE(16, 34); header.write("data", 36, "ascii"); header.writeUInt32LE(data.length, 40);
  return Buffer.concat([header, data]).toString("base64");
}
export const AUDIO_ROWS: Row[] = Array.from({ length: 30 }, (_, i) => ({ x: i % 2 ? wav(200 + 15 * (i % 7)) : wav(2500 + 40 * (i % 7)), label: i % 2 ? "low" : "high" }));

/** A deterministic hourly AR(1)-like series (timestamp + value) for forecasting; no randomness, so every runtime agrees. */
export const SERIES_ROWS: Row[] = (() => {
  const rows: Row[] = []; let y = 0;
  for (let i = 0; i < 160; i++) { y = 0.8 * y + Math.sin(i * 1.7) + 0.5 * Math.cos(i * 0.31); rows.push({ at: new Date(Date.UTC(2025, 0, 1) + i * 3_600_000).toISOString(), y }); }
  return rows;
})();

/** Tabular rows with planted far outliers; `outlier` is the boolean label (anomaly = true), never a feature. */
export const ANOMALY_ROWS: Row[] = Array.from({ length: 120 }, (_, i) => {
  const planted = i % 15 === 0;
  return { a: planted ? 8 * Math.sign(Math.sin(i)) : Math.sin(i * 1.3), b: planted ? -8 : Math.cos(i * 0.7), outlier: planted };
});
