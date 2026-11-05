import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from config import (
    MAX_WORKERS, SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS, TASK_CLEAR_AFTER_CREAT_SECONDS
)
from .compute_service import start_compute


class TaskStatus(Enum):
    """Task status enumeration"""
    PENDING = "pending"  # Waiting to be processed
    RUNNING = "running"  # Currently being processed
    COMPLETED = "completed"  # Processing completed
    FAILED = "failed"  # Processing failed


class Task:  # pylint: disable=too-few-public-methods

    def __init__(self, task_id: str, operation_type: str, data: Dict[str, Any]) -> None:
        self.task_id = task_id
        self.operation_type = operation_type
        self.data = data
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            'task_id': self.task_id,
            'operation_type': self.operation_type,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }


class TaskManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize task manager"""
        if not hasattr(self, '_initialized'):
            self._tasks = {}
            # Re-entrant: terminal-state writes may signal a task event while
            # already holding the lock.
            self._task_lock = threading.RLock()
            self._initialized = True
            self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            self._semaphore = threading.Semaphore(MAX_WORKERS)
            self._task_events: Dict[str, threading.Event] = {}
            self._start_cleanup_thread()

    def create_task(self, operation_type: str, data: Dict[str, Any],
                    timeout_seconds: Optional[float] = None) -> str:
        """Create the new task and return task ID.

        ``timeout_seconds`` overrides the default
        ``SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS`` for this single task; the
        method-tree op_type uses this so long enumerations are not killed
        by the global timeout.
        """
        task_id = str(uuid.uuid4())
        task = Task(task_id, operation_type, data)
        task.timeout_seconds = timeout_seconds

        with self._task_lock:
            self._tasks[task_id] = task
            self._task_events[task_id] = threading.Event()

        data['task_id'] = task_id

        thread = threading.Thread(target=self._execute_task, args=(task,))
        thread.daemon = True
        thread.start()

        return task_id

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        with self._task_lock:
            task = self._tasks.get(task_id)
            if task:
                return task.to_dict()
            return None

    def wait_for_task(self, task_id: str, timeout: float = 2.0) -> Optional[Dict[str, Any]]:
        """Block until task completes or timeout elapses, then return current status.

        Returns None if the task no longer exists.
        """
        with self._task_lock:
            task = self._tasks.get(task_id)
            event = self._task_events.get(task_id)
        if task is None:
            return None
        if event is not None:
            event.wait(timeout)
        with self._task_lock:
            task = self._tasks.get(task_id)
            return task.to_dict() if task else None

    def _set_final_state(self, task: Task, status: TaskStatus,
                         result=None, error=None) -> None:
        """Publish a terminal state atomically.

        The payload is written *before* the status: an SSE stream that sees
        ``completed`` must never observe ``result is None``.
        """
        with self._task_lock:
            if result is not None:
                task.result = result
            if error is not None:
                task.error = error
            task.completed_at = datetime.now()
            task.status = status

    def _execute_task(self, task: Task):
        """Execute the task (runs in background thread)"""
        timeout = (task.timeout_seconds
                   if getattr(task, 'timeout_seconds', None)
                   else SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS)

        # Acquire a worker slot. A task that cannot get one within its own budget
        # is failed instead of waiting forever (hung tasks keep their slot until
        # they really finish, so this is honest backpressure, not a false error).
        if not self._semaphore.acquire(timeout=timeout):  # pylint: disable=consider-using-with
            self._set_final_state(task, TaskStatus.FAILED,
                                  error="服务正忙, 任务排队超时, 请稍后重试.")
            self._set_task_event(task.task_id)
            return

        # Normally the ``finally`` below releases the slot; a timed-out task that
        # keeps computing in the executor stays counted until it actually ends.
        release_semaphore = True
        try:
            with self._task_lock:
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

            future = self._executor.submit(
                start_compute, task.operation_type, task.data)
            try:
                success, result = future.result(timeout=timeout)
                if success:
                    self._set_final_state(task, TaskStatus.COMPLETED, result=result)
                else:
                    self._set_final_state(task, TaskStatus.FAILED, error=result)
            except TimeoutError:
                self._set_final_state(task, TaskStatus.FAILED, error=(
                    "任务超时, 程序无法求解该题. 若确定其有初等函数解, "
                    "可尝试等价变形表达式后再次运行计算."))
                # The computation cannot be cancelled (SymPy has no kill switch).
                # Keep its slot occupied until the future finishes, otherwise
                # timed-out work would let the executor queue grow without bound
                # and every later task would time out behind it.
                release_semaphore = False
                self._release_semaphore_when_done(future)

        except Exception as e:
            self._set_final_state(task, TaskStatus.FAILED, error=str(e))

        finally:
            if release_semaphore:
                self._semaphore.release()
            self._set_task_event(task.task_id)

    def _release_semaphore_when_done(self, future) -> None:
        """Release the worker slot only once an abandoned computation finishes."""
        def _waiter():
            try:
                future.result()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            finally:
                self._semaphore.release()

        threading.Thread(target=_waiter, daemon=True).start()

    def _set_task_event(self, task_id: str) -> None:
        """Signal any waiter (SSE/polling) that the task reached a final state."""
        with self._task_lock:
            event = self._task_events.get(task_id)
            if event:
                event.set()

    def _start_cleanup_thread(self):
        def cleanup_worker():
            while True:
                time.sleep(30)  # Cleanup every 30 seconds
                self.cleanup_old_tasks()

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def cleanup_old_tasks(self, max_age_seconds: int = TASK_CLEAR_AFTER_CREAT_SECONDS):
        """Clean up finished tasks once their result has had time to be read.

        The TTL starts at ``completed_at``, not at creation: a method-tree run
        can legitimately take 610s/7200s, and a task waiting for a worker slot
        (PENDING) may wait minutes behind a timed-out SymPy computation.
        Evicting either by creation time silently lost the user's result.
        """
        current_time = datetime.now()
        with self._task_lock:
            tasks_to_delete = []
            for task_id, task in self._tasks.items():
                if task.status in (TaskStatus.PENDING, TaskStatus.RUNNING):
                    continue
                reference = task.completed_at or task.created_at
                if (current_time - reference).total_seconds() > max_age_seconds:
                    tasks_to_delete.append(task_id)

            for task_id in tasks_to_delete:
                self._task_events.pop(task_id, None)
                del self._tasks[task_id]


# Global task manager instance
task_manager = TaskManager()
