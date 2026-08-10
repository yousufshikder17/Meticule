import type pg from "pg";
import { calculatorTool } from "./calculator.js";
import { noteStorageTool, readOnlySqlTool } from "./database-tools.js";
import { memoryStoreTool } from "./memory-tool.js";
import { createKnowledgeSearchTool } from "./retrieval-tool.js";
import type { RetrievalService } from "../retrieval/retrieval-service.js";
import type { ConnectorService } from "../connectors/connector-service.js";
import { createMcpCallTool } from "./mcp-tool.js";
import { ToolRegistry } from "./types.js";
import { OrchestrationService } from "../orchestration/orchestration-service.js";
import { createDelegateRunTool } from "./delegation-tool.js";

export function createToolRegistry(_pool: pg.Pool, retrieval?: RetrievalService | null, connectors?: ConnectorService | null): ToolRegistry {
  const registry = new ToolRegistry();
  registry.register(calculatorTool); registry.register(noteStorageTool); registry.register(readOnlySqlTool); registry.register(memoryStoreTool);
  registry.register(createDelegateRunTool(new OrchestrationService(_pool)));
  if (retrieval) registry.register(createKnowledgeSearchTool(retrieval));
  if (connectors) registry.register(createMcpCallTool(connectors));
  return registry;
}
