import { isIP } from "node:net";
import { lookup as systemLookup } from "node:dns/promises";

export class ConnectorNetworkPolicyError extends Error {}
type AddressLookup = (hostname: string) => Promise<{ address: string; family: number }[]>;

function ipv4Private(address: string): boolean {
  const parts = address.split(".").map(Number); if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return true;
  const [a, b, c] = parts as [number, number, number, number];
  return a === 0 || a === 10 || a === 127 || a >= 224
    || (a === 100 && b >= 64 && b <= 127)
    || (a === 169 && b === 254)
    || (a === 172 && b >= 16 && b <= 31)
    || (a === 192 && ((b === 0 && [0, 2].includes(c)) || (b === 88 && c === 99) || b === 168))
    || (a === 198 && ([18, 19].includes(b) || (b === 51 && c === 100)))
    || (a === 203 && b === 0 && c === 113);
}
function privateAddress(address: string): boolean {
  const family = isIP(address); if (family === 4) return ipv4Private(address);
  if (family !== 6) return true;
  const normalized = address.toLowerCase();
  if (normalized.startsWith("::ffff:")) return ipv4Private(normalized.slice(7));
  return normalized === "::" || normalized === "::1" || normalized.startsWith("fc") || normalized.startsWith("fd") || normalized.startsWith("fe8") || normalized.startsWith("fe9") || normalized.startsWith("fea") || normalized.startsWith("feb") || normalized.startsWith("ff");
}

export class ConnectorNetworkPolicy {
  private readonly origins: Set<string>;
  constructor(allowedOrigins: string[], private readonly allowPrivateNetwork = false, private readonly lookup: AddressLookup = async (hostname) => systemLookup(hostname, { all: true, verbatim: true })) {
    if (!allowedOrigins.length) throw new ConnectorNetworkPolicyError("At least one exact connector egress origin is required");
    this.origins = new Set(allowedOrigins.map((entry) => {
      const url = new URL(entry); if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.pathname !== "/" || url.search || url.hash) throw new ConnectorNetworkPolicyError("Connector egress entries must be exact HTTP(S) origins");
      return url.origin.toLowerCase();
    }));
  }

  async assertAllowed(rawUrl: string): Promise<URL> {
    const url = new URL(rawUrl);
    if (!["http:", "https:"].includes(url.protocol)) throw new ConnectorNetworkPolicyError("Connector URL must use HTTP or HTTPS");
    if (url.username || url.password || url.hash) throw new ConnectorNetworkPolicyError("Connector URL cannot contain userinfo or a fragment");
    if (!this.origins.has(url.origin.toLowerCase())) throw new ConnectorNetworkPolicyError("Connector origin is not allowlisted");
    const addresses = isIP(url.hostname) ? [{ address: url.hostname, family: isIP(url.hostname) }] : await this.lookup(url.hostname);
    if (!addresses.length) throw new ConnectorNetworkPolicyError("Connector hostname did not resolve");
    if (!this.allowPrivateNetwork && addresses.some(({ address }) => privateAddress(address))) throw new ConnectorNetworkPolicyError("Connector hostname resolves to a private or special-use address");
    return url;
  }
}
