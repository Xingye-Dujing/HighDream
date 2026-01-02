import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime
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


class Task:

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
            self._task_lock = threading.Lock()
            self._initialized = True
            self._executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            # Semaphore to control concurrent tasks
            self._semaphore = threading.Semaphore(MAX_WORKERS)
            self._start_cleanup_thread()

    def create_task(self, operation_type: str, data: Dict[str, Any]) -> str:
        """Create new task and return task ID"""
        task_id = str(uuid.uuid4())
        task = Task(task_id, operation_type, data)

        with self._task_lock:
            self._tasks[task_id] = task

        # Start asynchronous processing
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

    def _execute_task(self, task: Task):
        """Execute task (runs in background thread)"""
        # Acquire semaphore to ensure we don't exceed max workers
        with self._semaphore:
            try:
                # Update status to running
                task.status = TaskStatus.RUNNING
                task.started_at = datetime.now()

                future = self._executor.submit(
                    start_compute, task.operation_type, task.data)
                try:
                    success, result = future.result(
                        timeout=SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS)
                    # Update task result
                    if success:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = result
                except TimeoutError:
                    task.status = TaskStatus.FAILED
                    task.error = "任务超时, 程序无法求解该题. 若确定其有初等函数解, 可尝试等价变形表达式后再次运行计算."

            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)

            finally:
                task.completed_at = datetime.now()

    def _start_cleanup_thread(self):
        def cleanup_worker():
            while True:
                time.sleep(30)  # Cleanup every 30 seconds
                self.cleanup_old_tasks()

        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

    def cleanup_old_tasks(self, max_age_seconds: int = TASK_CLEAR_AFTER_CREAT_SECONDS):
        """Clean up old tasks"""
        current_time = datetime.now()
        with self._task_lock:
            # Collect task IDs to delete first to avoid modifying dict during iteration
            tasks_to_delete = []
            for task_id, task in self._tasks.items():
                if (current_time - task.created_at).total_seconds() > max_age_seconds:
                    tasks_to_delete.append(task_id)

            for task_id in tasks_to_delete:
                del self._tasks[task_id]


# Global task manager instance
task_manager = TaskManager()
