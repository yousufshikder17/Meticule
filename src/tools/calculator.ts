import { z } from "zod";
import type { ToolDefinition } from "./types.js";

function calculate(source: string): number {
  const tokens = source.match(/\d+(?:\.\d+)?|[()+\-*/]/g);
  if (!tokens || tokens.join("") !== source.replace(/\s/g, "")) throw new Error("Expression contains unsupported characters");
  let position = 0;
  const expression = (): number => { let value = term(); while (["+", "-"].includes(tokens[position] ?? "")) { const op=tokens[position++]; const right=term(); value=op==="+"?value+right:value-right; } return value; };
  const term = (): number => { let value = factor(); while (["*", "/"].includes(tokens[position] ?? "")) { const op=tokens[position++]; const right=factor(); if(op==="/"&&right===0) throw new Error("Division by zero"); value=op==="*"?value*right:value/right; } return value; };
  const factor = (): number => { const token=tokens[position++]; if(token==="-") return -factor(); if(token==="("){const value=expression(); if(tokens[position++]!==")") throw new Error("Unclosed parenthesis"); return value;} const value=Number(token); if(!Number.isFinite(value)) throw new Error("Expected a number"); return value; };
  const result=expression(); if(position!==tokens.length || !Number.isFinite(result)) throw new Error("Invalid expression"); return result;
}

export const calculatorTool: ToolDefinition = {
  name: "calculator", description: "Evaluate finite arithmetic with +, -, *, /, and parentheses.",
  inputSchema: z.object({ expression: z.string().min(1).max(1000).regex(/^[\d\s()+\-*/.]+$/, "Expression contains unsupported characters") }),
  outputSchema: z.object({ value: z.number().finite() }), riskLevel: "low",
  authorization: { requiredRoles: [] }, approvalRequirement: "never", timeoutMs: 1000,
  retryPolicy: { maxAttempts: 1, retryableErrors: [] }, idempotency: "pure",
  retrySafety: "pure",
  async execute(input) { return { value: calculate((input as { expression: string }).expression) }; },
};
