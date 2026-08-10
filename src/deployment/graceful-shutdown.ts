export interface ClosableServer { close(callback: (error?: Error) => void): void; closeIdleConnections?: () => void; closeAllConnections?: () => void }

export async function closeServer(server: ClosableServer, graceMs: number): Promise<"drained" | "forced"> {
  server.closeIdleConnections?.();
  let timer: NodeJS.Timeout | undefined;
  const result = await Promise.race([
    new Promise<"drained">((resolve, reject) => server.close((error) => error ? reject(error) : resolve("drained"))),
    new Promise<"forced">((resolve) => { timer = setTimeout(() => { server.closeAllConnections?.(); resolve("forced"); }, graceMs); }),
  ]);
  if (timer) clearTimeout(timer); return result;
}
