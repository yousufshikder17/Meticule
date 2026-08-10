export class ConnectorCredentialError extends Error {}

export interface ConnectorCredentialResolver { resolve(reference: string | null): string | null }

export class EnvironmentConnectorCredentialResolver implements ConnectorCredentialResolver {
  constructor(private readonly env: NodeJS.ProcessEnv = process.env) {}
  resolve(reference: string | null): string | null {
    if (reference === null) return null;
    if (!/^CONNECTOR_SECRET_[A-Z0-9_]+$/.test(reference)) throw new ConnectorCredentialError("Connector credential reference is invalid");
    const secret = this.env[reference]; if (!secret) throw new ConnectorCredentialError(`Connector credential is unavailable for reference ${reference}`);
    return secret;
  }
}
