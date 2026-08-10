import { describe, expect, it } from "vitest";
import { deriveTaskStates, validatePlanGraph } from "../../src/planning/plan-schema.js";

describe("durable plan validation", () => {
  it("orders readiness and propagates failed dependencies", () => {
    const plan = deriveTaskStates(validatePlanGraph({ objective: "ship", tasks: [
      { taskId: "a", objective: "first", status: "completed", result: "done", evidenceStepId: "11111111-1111-4111-8111-111111111111" },
      { taskId: "b", objective: "second", dependencies: ["a"] },
      { taskId: "c", objective: "failed", status: "failed", error: "no", evidenceStepId: "22222222-2222-4222-8222-222222222222" },
      { taskId: "d", objective: "blocked", dependencies: ["c"] },
      { taskId: "e", objective: "transitively blocked", dependencies: ["d"] },
    ] }));
    expect(Object.fromEntries(plan.tasks.map((task) => [task.taskId, task.status]))).toEqual({ a: "completed", b: "ready", c: "failed", d: "blocked", e: "blocked" });
  });

  it("rejects duplicate tasks, missing dependencies, and cycles", () => {
    expect(() => validatePlanGraph({ objective: "x", tasks: [{ taskId: "a", objective: "x" }, { taskId: "a", objective: "y" }] })).toThrow(/Duplicate/);
    expect(() => validatePlanGraph({ objective: "x", tasks: [{ taskId: "a", objective: "x", dependencies: ["missing"] }] })).toThrow(/Unknown/);
    expect(() => validatePlanGraph({ objective: "x", tasks: [{ taskId: "a", objective: "x", dependencies: ["b"] }, { taskId: "b", objective: "y", dependencies: ["a"] }] })).toThrow(/cycle/);
  });
});
