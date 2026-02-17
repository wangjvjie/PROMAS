from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Callable

from .models import CodeTask, TaskStatus, TaskWaiting, WorkerResult


class ParallelTaskScheduler:
    """Dependency-aware scheduler for parallel code generation.

    The scheduler honors explicit dependencies from planning and guarantees that
    the same file is never executed concurrently by multiple workers.
    """

    def __init__(
        self,
        tasks: list[CodeTask],
        *,
        worker_count: int,
        agent_factory: Callable[[str], object],
        max_attempts: int = 3,
        on_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.tasks = {t.task_id: t for t in tasks}
        self.worker_count = max(1, worker_count)
        self.agent_factory = agent_factory
        self.max_attempts = max_attempts
        self.on_event = on_event
        self.results: dict[str, WorkerResult] = {}

    def run(self) -> dict[str, WorkerResult]:
        worker_ids = [f"worker-{i+1}" for i in range(self.worker_count)]
        agents = {wid: self.agent_factory(wid) for wid in worker_ids}
        idle_workers = deque(worker_ids)
        in_flight: dict[Future, tuple[str, str]] = {}

        with ThreadPoolExecutor(max_workers=self.worker_count) as pool:
            while not self._all_terminal():
                self._promote_failed_on_failed_dependencies()

                while idle_workers:
                    excluded = {t for t, _ in in_flight.values()}
                    active_files = {self.tasks[t].file for t in excluded if t in self.tasks}
                    ready_tasks = self._ready_tasks(exclude=excluded, active_files=active_files)
                    if not ready_tasks:
                        break
                    task = ready_tasks[0]
                    worker_id = idle_workers.popleft()
                    task.status = TaskStatus.RUNNING
                    task.assigned_worker = worker_id
                    task.attempts += 1
                    self._emit(
                        {
                            "type": "task_assigned",
                            "task_id": task.task_id,
                            "file": task.file,
                            "worker": worker_id,
                            "attempt": task.attempts,
                            "mode": task.mode,
                            "source": task.source,
                        }
                    )

                    dep_files = [self.tasks[d].file for d in sorted(task.depends_on) if d in self.tasks]
                    completed_files = [t.file for t in self.tasks.values() if t.status == TaskStatus.DONE]
                    agent = agents[worker_id]

                    fut = pool.submit(
                        agent.execute,
                        task,
                        dependency_files=dep_files,
                        completed_files=completed_files,
                    )
                    in_flight[fut] = (task.task_id, worker_id)

                if not in_flight:
                    # nothing runnable and nothing running: deadlock or terminal
                    if not self._all_terminal():
                        self._mark_blocked_as_failed("deadlock: unresolved dependencies")
                        self._emit({"type": "scheduler_deadlock"})
                    continue

                done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    task_id, worker_id = in_flight.pop(fut)
                    idle_workers.append(worker_id)
                    task = self.tasks[task_id]

                    try:
                        result = fut.result()
                    except TaskWaiting as tw:
                        task.status = TaskStatus.BLOCKED
                        task.wait_reason = tw.reason
                        self._emit(
                            {
                                "type": "task_waiting",
                                "task_id": task.task_id,
                                "file": task.file,
                                "worker": worker_id,
                                "reason": tw.reason,
                            }
                        )
                        if task.attempts >= self.max_attempts:
                            task.status = TaskStatus.FAILED
                            task.last_error = f"max retries reached while waiting: {tw.reason}"
                            self._emit(
                                {
                                    "type": "task_failed",
                                    "task_id": task.task_id,
                                    "file": task.file,
                                    "reason": task.last_error,
                                }
                            )
                    except Exception as exc:
                        task.last_error = str(exc)
                        if task.attempts >= self.max_attempts:
                            task.status = TaskStatus.FAILED
                            self._emit(
                                {
                                    "type": "task_failed",
                                    "task_id": task.task_id,
                                    "file": task.file,
                                    "reason": task.last_error,
                                }
                            )
                        else:
                            task.status = TaskStatus.BLOCKED
                            self._emit(
                                {
                                    "type": "task_blocked",
                                    "task_id": task.task_id,
                                    "file": task.file,
                                    "reason": task.last_error,
                                }
                            )
                    else:
                        task.status = result.status
                        task.wait_reason = result.wait_reason
                        self.results[task_id] = result

                        if result.status != TaskStatus.DONE and task.attempts < self.max_attempts:
                            task.status = TaskStatus.BLOCKED
                            self._emit(
                                {
                                    "type": "task_blocked",
                                    "task_id": task.task_id,
                                    "file": task.file,
                                    "reason": result.summary or result.wait_reason or "non-done result",
                                }
                            )
                        elif result.status != TaskStatus.DONE:
                            task.status = TaskStatus.FAILED
                            self._emit(
                                {
                                    "type": "task_failed",
                                    "task_id": task.task_id,
                                    "file": task.file,
                                    "reason": result.summary or "worker returned non-done status",
                                }
                            )
                        else:
                            self._emit(
                                {
                                    "type": "task_done",
                                    "task_id": task.task_id,
                                    "file": task.file,
                                    "worker": worker_id,
                                    "produced_file": result.produced_file,
                                    "summary": result.summary,
                                }
                            )

        return self.results

    def _all_terminal(self) -> bool:
        terminal = {TaskStatus.DONE, TaskStatus.FAILED}
        return all(t.status in terminal for t in self.tasks.values())

    def _ready_tasks(self, *, exclude: set[str], active_files: set[str]) -> list[CodeTask]:
        ready: list[CodeTask] = []
        for task in self.tasks.values():
            if task.task_id in exclude:
                continue
            if task.file in active_files:
                continue
            if task.status not in {TaskStatus.PENDING, TaskStatus.BLOCKED}:
                continue
            if task.attempts >= self.max_attempts:
                task.status = TaskStatus.FAILED
                task.last_error = "max retries reached"
                continue
            deps_done = all(
                dep in self.tasks and self.tasks[dep].status == TaskStatus.DONE
                for dep in task.depends_on
            )
            if deps_done:
                ready.append(task)
        ready.sort(key=lambda t: (t.priority, t.task_id))
        return ready

    def _promote_failed_on_failed_dependencies(self) -> None:
        for task in self.tasks.values():
            if task.status in {TaskStatus.DONE, TaskStatus.FAILED}:
                continue
            failed_deps = [d for d in task.depends_on if d in self.tasks and self.tasks[d].status == TaskStatus.FAILED]
            if failed_deps:
                task.status = TaskStatus.FAILED
                task.last_error = f"dependency failed: {', '.join(failed_deps)}"
                self._emit(
                    {
                        "type": "task_failed",
                        "task_id": task.task_id,
                        "file": task.file,
                        "reason": task.last_error,
                    }
                )

    def _mark_blocked_as_failed(self, reason: str) -> None:
        for task in self.tasks.values():
            if task.status in {TaskStatus.PENDING, TaskStatus.BLOCKED, TaskStatus.RUNNING}:
                task.status = TaskStatus.FAILED
                if not task.last_error:
                    task.last_error = reason
                self._emit(
                    {
                        "type": "task_failed",
                        "task_id": task.task_id,
                        "file": task.file,
                        "reason": task.last_error,
                    }
                )

    def _emit(self, event: dict) -> None:
        if self.on_event is None:
            return
        try:
            self.on_event(event)
        except Exception:
            pass
