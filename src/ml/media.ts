/**
 * Strict validation of media and timestamp cells (spec sec. 2, amendments C1-C5). Pure TypeScript, no dependencies.
 * ml/media_common.py and epoch_micros in ml/ml_runtime_common.py apply the same rules; the cases both sides must
 * decide alike are in tests/fixtures/media-vectors.json. Refusals are ConflictErrors with fixed messages that never quote
 * the cell.
 */
import { ConflictError } from "../domain/errors.js";

// The same numbers as the vector file's `limits` block (the media unit test compares them).
export const MEDIA_LIMITS = {
  mediaMaxBytes: 2 * 1024 * 1024,
  maxBase64Chars: 2_796_204, // 4 * ceil(2 MiB / 3): the base64 length of 2 MiB
  maxSide: 4096,
  maxPixels: 4096 * 4096,
  maxChunks: 10_000, // chunks or segments one header walk visits, so a crafted file cannot buy unbounded work
  minAudioSeconds: 0.1,
  maxAudioSeconds: 30,
  minYear: 1900,
  maxYear: 2199,
} as const;
// Aggregate media one request may carry (C5). The /ml body limit (8,000,000 bytes) stays the effective dataset bound.
export const MEDIA_BUDGETS = {
  ingest: { pixels: 268_435_456, audioSeconds: 600 },
  prediction: { pixels: 67_108_864, audioSeconds: 120 },
} as const;

function refuse(message: string): never { throw new ConflictError(message); }

const BASE64_BODY = /^[A-Za-z0-9+/]*$/;
/**
 * Bytes of a media cell in standard base64 by the C1 grammar ^(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?$
 * (no data: prefix, URL-safe alphabet, whitespace or inner padding), at most 2 MiB decoded. Checked as "length a multiple
 * of 4, then the alphabet up to one or two final '='", which is equivalent and needs no backtracking on 2.8 MB cells.
 */
export function decodeBase64(text: string): Buffer {
  if (text.length > MEDIA_LIMITS.maxBase64Chars || text.length % 4) refuse("Media cell is not valid base64");
  const body = text.endsWith("==") ? text.slice(0, -2) : text.endsWith("=") ? text.slice(0, -1) : text;
  if (!BASE64_BODY.test(body)) refuse("Media cell is not valid base64");
  if ((text.length / 4) * 3 - (text.length - body.length) > MEDIA_LIMITS.mediaMaxBytes) refuse("Media cell exceeds 2 MiB");
  return Buffer.from(text, "base64");
}

const PNG_MAGIC = Buffer.from("89504e470d0a1a0a", "hex"), JPEG_MAGIC = Buffer.from("ffd8ff", "hex");
const PNG_IHDR = Buffer.from("0000000d49484452", "hex"); // length 13, then "IHDR"
// Start-of-frame markers that carry the image size (C4, C8 and CC are DHT, JPG and DAC).
const JPEG_SOF = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
export interface ImageHeader { format: "png" | "jpeg"; width: number; height: number }

/** IHDR must be the first chunk (length 13); every chunk lies inside the buffer; the walk ends at IEND (bytes after it are never read) and refuses any later IHDR. */
function pngSize(b: Buffer): [width: number, height: number] {
  if (b.length < 33 || !b.subarray(8, 16).equals(PNG_IHDR)) refuse("Malformed PNG");
  for (let pos = 8, chunks = 1; ; chunks++) {
    if (chunks > MEDIA_LIMITS.maxChunks || b.length - pos < 12) refuse("Malformed PNG");
    const length = b.readUInt32BE(pos), kind = b.toString("latin1", pos + 4, pos + 8);
    if (length > b.length - pos - 12 || (kind === "IHDR" && pos !== 8)) refuse("Malformed PNG");
    if (kind === "IEND") return [b.readUInt32BE(16), b.readUInt32BE(20)];
    pos += 12 + length;
  }
}

/**
 * Walks marker segments from SOI to the first SOS. 0xFF fill bytes may precede a marker; every other marker before SOS
 * must carry a length (>= 2) that stays inside the buffer. Exactly one SOFn, before SOS, whose length is
 * 8 + 3 * components (1..4); a second SOFn (a size that could differ from the one checked) is refused.
 */
function jpegSize(b: Buffer): [width: number, height: number] {
  let pos = 2, size: [number, number] | undefined;
  for (let segments = 1; ; segments++) {
    while (b.length - pos >= 2 && b[pos] === 0xff && b[pos + 1] === 0xff) pos++;
    if (segments > MEDIA_LIMITS.maxChunks || b.length - pos < 4 || b[pos] !== 0xff) refuse("Malformed JPEG");
    const marker = b[pos + 1]!, length = b.readUInt16BE(pos + 2);
    if (marker === 0x00 || marker === 0x01 || marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7) || length < 2 || length > b.length - pos - 2) refuse("Malformed JPEG");
    if (JPEG_SOF.has(marker)) {
      const components = length >= 8 ? b[pos + 9]! : 0;
      if (size || components < 1 || components > 4 || length !== 8 + 3 * components) refuse("Malformed JPEG");
      size = [b.readUInt16BE(pos + 7), b.readUInt16BE(pos + 5)];
    } else if (marker === 0xda) {
      return size ?? refuse("Malformed JPEG");
    }
    pos += 2 + length;
  }
}

/** Format and size of a PNG or JPEG from its header walk (C2): each side 1..4096, at most 16,777,216 pixels. Pixels are never decoded here. */
export function imageHeader(bytes: Buffer): ImageHeader {
  const format = bytes.subarray(0, 8).equals(PNG_MAGIC) ? "png" : bytes.subarray(0, 3).equals(JPEG_MAGIC) ? "jpeg" : refuse("Image must be PNG or JPEG");
  const [width, height] = format === "png" ? pngSize(bytes) : jpegSize(bytes);
  const { maxSide, maxPixels } = MEDIA_LIMITS;
  if (!(width >= 1 && width <= maxSide && height >= 1 && height <= maxSide && width * height <= maxPixels)) refuse("Image dimensions out of bounds");
  return { format, width, height };
}

const PCM = 1, IEEE_FLOAT = 3, EXTENSIBLE = 0xfffe;
// KSDATAFORMAT_SUBTYPE_PCM / _IEEE_FLOAT are {0000000X-0000-0010-8000-00AA00389B71}: two format bytes, then this.
const GUID_TAIL = Buffer.from("000000001000800000aa00389b71", "hex");
export interface WavLayout { format: "pcm" | "float"; bits: number; channels: number; rate: number; frames: number; offset: number }

/**
 * The C3 walk. RIFF size == byte length - 8; every chunk lies inside the buffer and an odd-sized one is followed by a pad
 * byte (absent only at the very end); other chunk types are skipped; exactly one `fmt ` before exactly one `data`; PCM
 * 8/16/24/32-bit or IEEE float 32-bit, directly or as WAVE_FORMAT_EXTENSIBLE (cbSize 22, SubFormat PCM/IEEE_FLOAT,
 * validBits == bits); 1-2 channels; 8000-48000 Hz; blockAlign == channels * bits / 8; byteRate == rate * blockAlign;
 * data a whole number of frames lasting 0.1 to 30 seconds (NaN-safe); float samples finite.
 */
export function wavLayout(b: Buffer): WavLayout {
  if (b.length < 12 || b.toString("latin1", 0, 4) !== "RIFF" || b.toString("latin1", 8, 12) !== "WAVE" || b.readUInt32LE(4) !== b.length - 8) refuse("Malformed WAV");
  let fmt: Buffer | undefined, body: { offset: number; size: number } | undefined;
  for (let pos = 12, chunks = 1; pos < b.length; chunks++) {
    if (chunks > MEDIA_LIMITS.maxChunks || b.length - pos < 8) refuse("Malformed WAV");
    const kind = b.toString("latin1", pos, pos + 4), size = b.readUInt32LE(pos + 4);
    if (size > b.length - pos - 8) refuse("Malformed WAV");
    if (kind === "fmt ") {
      if (fmt || body) refuse("Malformed WAV");
      fmt = b.subarray(pos + 8, pos + 8 + size);
    } else if (kind === "data") {
      if (!fmt || body) refuse("Malformed WAV");
      body = { offset: pos + 8, size };
    }
    pos += 8 + size + (size % 2);
  }
  if (!fmt || !body || fmt.length < 16) refuse("Malformed WAV");
  let tag = fmt.readUInt16LE(0);
  const channels = fmt.readUInt16LE(2), rate = fmt.readUInt32LE(4), byteRate = fmt.readUInt32LE(8), blockAlign = fmt.readUInt16LE(12), bits = fmt.readUInt16LE(14);
  if (tag === EXTENSIBLE) {
    if (fmt.length < 40 || fmt.readUInt16LE(16) !== 22 || fmt.readUInt16LE(18) !== bits || !fmt.subarray(26, 40).equals(GUID_TAIL) || fmt[25] !== 0 || (fmt[24] !== PCM && fmt[24] !== IEEE_FLOAT)) refuse("Unsupported WAV format");
    tag = fmt[24]!;
  }
  if (!((tag === PCM && [8, 16, 24, 32].includes(bits)) || (tag === IEEE_FLOAT && bits === 32))) refuse("Unsupported WAV format");
  if (!(channels >= 1 && channels <= 2 && rate >= 8000 && rate <= 48000)) refuse("Unsupported WAV format");
  if (blockAlign !== (channels * bits) / 8 || byteRate !== rate * blockAlign || body.size % blockAlign) refuse("Malformed WAV");
  const frames = body.size / blockAlign, seconds = frames / rate;
  if (!(seconds >= MEDIA_LIMITS.minAudioSeconds && seconds <= MEDIA_LIMITS.maxAudioSeconds)) refuse("Audio duration out of bounds");
  if (tag === IEEE_FLOAT) for (let i = 0; i < frames * channels; i++) if (!Number.isFinite(b.readFloatLE(body.offset + 4 * i))) refuse("Audio samples must be finite");
  return { format: tag === IEEE_FLOAT ? "float" : "pcm", bits, channels, rate, frames, offset: body.offset };
}

/** Running media totals of one request (C5); a charge past either limit is refused. */
export class MediaBudget {
  private pixels = 0;
  private audioMicros = 0;
  constructor(private readonly limits: { readonly pixels: number; readonly audioSeconds: number }) {}
  image({ width, height }: ImageHeader): void {
    this.pixels += width * height;
    if (this.pixels > this.limits.pixels) refuse(`Images in one request exceed ${this.limits.pixels} pixels`);
  }
  audio({ frames, rate }: WavLayout): void {
    // Whole microseconds, rounded up per clip, so the running total is exact integer arithmetic.
    this.audioMicros += Math.ceil((frames * 1e6) / rate);
    if (this.audioMicros > this.limits.audioSeconds * 1e6) refuse(`Audio in one request exceeds ${this.limits.audioSeconds} seconds`);
  }
}

// ISO-8601 with an explicit zone (spec sec. 2). JavaScript's \d is ASCII-only, and $ never matches before a final newline.
const TIMESTAMP = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,6}))?)?(?:Z|([+-])(\d{2}):(\d{2}))$/;
const MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/**
 * UTC epoch milliseconds of a timestamp cell (a fractional part carries the microseconds), or undefined when it breaks a
 * C4 rule: the pattern above in at most 40 characters, a real proleptic Gregorian date, hour 0-23, minute and second
 * 0-59, offset at most 23:59, year 1900..2199. Never Date.parse. The value is one correctly rounded division of the exact
 * integer microseconds (< 2^53 for every allowed year), so distinct instants never tie or swap order.
 */
export function timestampMs(text: string): number | undefined {
  const m = text.length <= 40 ? TIMESTAMP.exec(text) : null;
  if (!m) return undefined;
  const [year, month, day, hour, minute, second, offsetHours, offsetMinutes] = [1, 2, 3, 4, 5, 6, 9, 10].map(i => Number(m[i] ?? 0)) as [number, number, number, number, number, number, number, number];
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  if (year < MEDIA_LIMITS.minYear || year > MEDIA_LIMITS.maxYear || month < 1 || month > 12 || day < 1 || day > (month === 2 && leap ? 29 : MONTH_DAYS[month - 1]!)
    || hour > 23 || minute > 59 || second > 59 || offsetHours > 23 || offsetMinutes > 59) return undefined;
  const offsetMs = (offsetHours * 60 + offsetMinutes) * 60_000 * (m[8] === "-" ? -1 : 1);
  return ((Date.UTC(year, month - 1, day, hour, minute, second) - offsetMs) * 1000 + Number((m[7] ?? "").padEnd(6, "0"))) / 1000;
}
