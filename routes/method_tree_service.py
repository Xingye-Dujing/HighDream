"""Registry that connects TaskManager task IDs to running MethodTreeEnumerators.

The :class:`~core.method_tree_enumerator.MethodTreeEnumerator` runs inside
a worker thread managed by :class:`routes.task_manager.TaskManager`. The
HTTP layer needs a way to:

* poll incremental progress while the worker is still computing, and
* cancel a running enumeration on user request.

This module keeps a small ``task_id -> enumerator`` map plus the
thread-safe helpers the API endpoints need. It is intentionally tiny and
does not duplicate any lifecycle logic — task start / timeout / cleanup
still lives in TaskManager.
"""

import threading
from typing import Any, Dict, Optional, Tuple

from core.method_tree_enumerator import MethodTreeEnumerator


_running: Dict[str, MethodTreeEnumerator] = {}
_lock = threading.Lock()


def _extract_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the subset of keys MethodTreeEnumerator accepts."""
    return {
        'domain': data.get('domain', 'diff'),
        'expression': data.get('expression', ''),
        'variable': data.get('variable', 'x'),
        'point': data.get('point', 0),
        'direction': data.get('direction', '+'),
        'max_depth': data.get('max_depth'),
        'max_nodes': data.get('max_nodes'),
        'time_limit_seconds': data.get('time_limit_seconds'),
    }


def start_method_tree(data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Entry point invoked by TaskManager's worker thread.

    ``data`` must include ``task_id`` so the enumerator can be registered
    for snapshot / cancel while it runs.
    """
    task_id = data.get('task_id')
    if not task_id:
        return False, "missing task_id"

    try:
        enumerator = MethodTreeEnumerator(**_extract_kwargs(data))
    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, f"初始化失败: {e}"

    with _lock:
        _running[task_id] = enumerator
    try:
        tree = enumerator.run()
    finally:
        with _lock:
            _running.pop(task_id, None)
    return True, tree


def snapshot_method_tree(task_id: str) -> Optional[Dict[str, Any]]:
    """Return a thread-safe snapshot of the in-progress tree (or None)."""
    with _lock:
        enumerator = _running.get(task_id)
    if enumerator is None:
        return None
    return enumerator.snapshot()


def cancel_method_tree(task_id: str) -> bool:
    """Request cancellation of a running enumerator. Returns True if found."""
    with _lock:
        enumerator = _running.get(task_id)
    if enumerator is None:
        return False
    enumerator.cancel()
    return True
