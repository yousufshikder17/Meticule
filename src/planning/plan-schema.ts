import { z } from "zod";

export const PlanTaskStatusSchema = z.enum(["pending", "ready", "running", "completed", "failed", "blocked"]);
export const PlanTaskSchema = z.object({
  taskId: z.string().trim().min(1).max(100).regex(/^[a-zA-Z0-9._-]+$/),
  objective: z.string().trim().min(1).max(2_000),
  dependencies: z.array(z.string().min(1).max(100)).max(50).default([]),
  status: PlanTaskStatusSchema.default("pending"),
  assignedExecutor: z.literal("native").default("native"),
  attemptCount: z.int().nonnegative().default(0),
  result: z.unknown().nullable().default(null),
  error: z.unknown().nullable().default(null),
  evidenceStepId: z.uuid().nullable().default(null),
});

export const PlanSchema = z.object({
  objective: z.string().trim().min(1).max(10_000),
  tasks: z.array(PlanTaskSchema).min(1).max(100),
});
export type Plan = z.infer<typeof PlanSchema>;

export function validatePlanGraph(input: unknown): Plan {
  const plan = PlanSchema.parse(input);
  const ids = new Set<string>();
  for (const task of plan.tasks) {
    if (ids.has(task.taskId)) throw new Error(`Duplicate plan task: ${task.taskId}`);
    ids.add(task.taskId);
  }
  for (const task of plan.tasks) for (const dependency of task.dependencies) {
    if (!ids.has(dependency)) throw new Error(`Unknown dependency ${dependency} for ${task.taskId}`);
    if (dependency === task.taskId) throw new Error(`Plan task ${task.taskId} cannot depend on itself`);
  }
  for (const task of plan.tasks) if (new Set(task.dependencies).size !== task.dependencies.length) throw new Error(`Duplicate dependency for ${task.taskId}`);
  const visiting = new Set<string>(); const visited = new Set<string>();
  const byId = new Map(plan.tasks.map((task) => [task.taskId, task]));
  const visit = (id: string) => {
    if (visiting.has(id)) throw new Error(`Plan dependency cycle includes ${id}`);
    if (visited.has(id)) return;
    visiting.add(id);
    for (const dependency of byId.get(id)!.dependencies) visit(dependency);
    visiting.delete(id); visited.add(id);
  };
  for (const id of ids) visit(id);
  return plan;
}

export function deriveTaskStates(plan: Plan): Plan {
  const tasks = plan.tasks.map((task) => ({ ...task, dependencies: [...task.dependencies] }));
  const byId = new Map(tasks.map((task) => [task.taskId, task]));
  let changed = true;
  while (changed) {
    changed = false;
    for (const task of tasks) {
      if (["completed", "failed", "running"].includes(task.status)) continue;
      const dependencies = task.dependencies.map((id) => byId.get(id)!);
      const next = dependencies.some((dependency) => ["failed", "blocked"].includes(dependency.status)) ? "blocked"
        : dependencies.every((dependency) => dependency.status === "completed") ? "ready" : "pending";
      if (task.status !== next) { task.status = next; changed = true; }
    }
  }
  return { ...plan, tasks };
}
