import { setTimeout as delay } from "node:timers/promises";
import { loadConfig } from "../config.js";
import { getPool, closePool } from "../db/pool.js";
import { LifecycleWorker } from "./worker.js";
import { createProviderRegistry } from "../models/provider-configuration.js";
import { ExecutionEngineRegistry } from "../execution/execution-engine-registry.js";
import { NativeExecutionEngine } from "../execution/native-engine.js";
import { createToolRegistry } from "../tools/registry.js";
import { AgentLoop } from "../execution/agent-loop.js";
import { ReconciliationService } from "../tools/reconciliation-service.js";
import { createRetrievalService } from "../retrieval/retrieval-configuration.js";
import { createConnectorService } from "../connectors/connector-configuration.js";
import { OrchestrationService } from "../orchestration/orchestration-service.js";
import { EvaluationService } from "../evaluation/evaluation-service.js";
import { OperationalService } from "../observability/operational-service.js";
import { logger } from "../observability/logger.js";
import { runSchedulerCycle } from "./scheduler-cycle.js";
import { mlConfiguration } from "../ml/configuration.js";
import { TrainingProcessor } from "../ml/training-processor.js";
import { PostgresArtifactStore } from "../ml/artifact-store.js";

const config = loadConfig();
const pool = getPool();
const providers = createProviderRegistry();
const retrieval = createRetrievalService(pool);
const connectors = createConnectorService(pool);
const engines = new ExecutionEngineRegistry(); engines.register(new NativeExecutionEngine());
const toolRegistry = createToolRegistry(pool, retrieval, connectors);
const loop = new AgentLoop(pool, providers, engines, toolRegistry, retrieval, connectors);
const reconciler = new ReconciliationService(pool, toolRegistry);
const orchestration = new OrchestrationService(pool);
const evaluations = new EvaluationService(pool);
const operations = new OperationalService(pool, providers);
const roles = config.WORKER_ROLES.split(",").map((role) => role.trim()).filter(Boolean);
const worker = new LifecycleWorker(pool, { workerId: config.WORKER_ID, leaseSeconds: config.LEASE_SECONDS }, { execute: (runId, workerId) => loop.execute(runId, workerId, roles) }, logger);
const ml = mlConfiguration();
const mlWorker = ml.enabled ? new LifecycleWorker(pool, { workerId: `${config.WORKER_ID}:ml`, leaseSeconds: config.LEASE_SECONDS, kind: "ml" }, new TrainingProcessor(pool, new PostgresArtifactStore(pool), ml.backends), logger) : null;
let stopping = false;
const requestStop = (): void => { stopping = true; void operations.heartbeat(config.WORKER_ID, "worker", { roles, pid: process.pid }, true).catch((error) => logger.log("error", "worker.drain_signal_failed", { workerId: config.WORKER_ID, error })); };
process.once("SIGINT", requestStop);
process.once("SIGTERM", requestStop);
await operations.heartbeat(config.WORKER_ID, "worker", { roles, pid: process.pid });

while (!stopping) {
  try {
    await operations.heartbeat(config.WORKER_ID, "worker", { roles, pid: process.pid });
    const progressed = await runSchedulerCycle([
      () => evaluations.advanceNext(`${config.WORKER_ID}:evaluator`),
      () => orchestration.advanceNext(`${config.WORKER_ID}:orchestrator`),
      () => reconciler.reconcileNext(`${config.WORKER_ID}:reconciler`, config.LEASE_SECONDS),
      () => worker.tick(),
      () => mlWorker?.tick() ?? Promise.resolve(false),
    ], () => stopping);
    if (!progressed && !stopping) await delay(config.WORKER_POLL_MS);
  }
  catch (error) { logger.log("error", "worker.tick_failed", { workerId: config.WORKER_ID, error }); await delay(config.WORKER_POLL_MS); }
}
await operations.heartbeat(config.WORKER_ID, "worker", { roles, pid: process.pid }, true);
await closePool();
