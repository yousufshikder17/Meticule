export type SchedulerTask = () => Promise<boolean>;

export async function runSchedulerCycle(tasks: SchedulerTask[], shouldStop: () => boolean): Promise<boolean> {
  let progressed = false;
  for (const task of tasks) { if (shouldStop()) break; progressed = await task() || progressed; }
  return progressed;
}
