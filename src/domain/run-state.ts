import { z } from "zod";

export const RUN_STATES = [
  "queued", "claimed", "running", "waiting_for_approval", "paused",
  "cancelling", "completed", "failed", "cancelled",
] as const;

export const RunStateSchema = z.enum(RUN_STATES);
export type RunState = z.infer<typeof RunStateSchema>;

export const TERMINAL_STATES = new Set<RunState>(["completed", "failed", "cancelled"]);

const transitions: Readonly<Record<RunState, ReadonlySet<RunState>>> = {
  queued: new Set(["claimed", "cancelling"]),
  claimed: new Set(["running", "queued", "cancelling"]),
  running: new Set(["waiting_for_approval", "paused", "cancelling", "completed", "failed"]),
  waiting_for_approval: new Set(["queued", "cancelling"]),
  paused: new Set(["queued", "cancelling"]),
  cancelling: new Set(["cancelled", "failed"]),
  completed: new Set(),
  failed: new Set(),
  cancelled: new Set(),
};

export class InvalidTransitionError extends Error {
  constructor(from: RunState, to: RunState) {
    super(`Invalid run transition: ${from} -> ${to}`);
    this.name = "InvalidTransitionError";
  }
}

export function canTransition(from: RunState, to: RunState): boolean {
  return transitions[from].has(to);
}

export function assertTransition(from: RunState, to: RunState): void {
  if (!canTransition(from, to)) throw new InvalidTransitionError(from, to);
}

export function isTerminal(state: RunState): boolean {
  return TERMINAL_STATES.has(state);
}
