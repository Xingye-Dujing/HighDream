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

import config
from core.input_validation import as_bool_flag
from core.method_tree_enumerator import MethodTreeEnumerator


_running: Dict[str, MethodTreeEnumerator] = {}
_lock = threading.Lock()


def _extract_kwargs(data: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the subset of keys MethodTreeEnumerator accepts."""
    interactive = as_bool_flag(data.get('interactive', False))
    kwargs = {
        'domain': data.get('domain', 'diff'),
        'expression': data.get('expression', ''),
        'variable': data.get('variable', 'x'),
        'point': data.get('point', 0),
        'direction': data.get('direction', '+'),
        'max_depth': data.get('max_depth'),
        'max_nodes': data.get('max_nodes'),
        'time_limit_seconds': 0 if interactive else data.get('time_limit_seconds'),
        'interactive': interactive,
    }
    return kwargs


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


def respond_method_tree(task_id: str, accepted: bool) -> bool:
    """Respond to a pending interactive decision. Returns True if responded."""
    with _lock:
        enumerator = _running.get(task_id)
    if enumerator is None:
        return False
    return enumerator.respond_decision(accepted)


def linear_path_method_tree(data: Dict[str, Any]) -> Dict[str, Any]:
    """Replay one root->node rule path as a single linear derivation (no re-search).

    The finished tree keeps only per-node LaTeX (its ``explanation`` is empty and its
    enumerator is dropped once the run completes), so we rebuild a fresh solver with the
    ORIGINAL problem config and re-apply just the chosen rule sequence — the same
    primitives ``_replay_solver`` used to build that branch — then emit the notebook-
    identical ``\\begin{align}`` block plus per-step ``{latex, explanation}``.
    """
    from core.manual_step_solver import ManualStepSolver  # pylint: disable=import-outside-toplevel

    kwargs = _extract_kwargs(data)
    rule_path = data.get('rule_path') or []
    if not isinstance(rule_path, list):
        return {'success': False, 'error': 'rule_path 必须是规则名数组'}
    # 'back_subs' is a synthetic node the enumerator invents, not a registry rule; the
    # back-substitution happens automatically in finish()/final_postprocess.
    steps_path = [r for r in rule_path if isinstance(r, str) and r != 'back_subs']
    steps_path = steps_path[:config.METHOD_TREE_HARD_MAX_DEPTH]

    try:
        solver = ManualStepSolver(
            domain=kwargs['domain'], expression=kwargs['expression'],
            variable=kwargs['variable'], point=kwargs['point'],
            direction=kwargs['direction'])
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {'success': False, 'error': f'初始化失败: {e}'}

    applied = 0
    for rule_name in steps_path:
        if solver.done:
            break
        try:
            state = solver.apply_rule(rule_name)
        except Exception:  # pylint: disable=broad-exception-caught
            break
        if state.get('error'):
            break
        applied += 1

    try:
        latex_block = solver.calculator.step_generator.to_latex()
        steps = solver.state()['steps']
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {'success': False, 'error': f'重放失败: {e}'}

    return {'success': True, 'domain': kwargs['domain'], 'node_id': data.get('node_id'),
            'applied': applied, 'requested': len(steps_path), 'done': solver.done,
            'result': latex_block, 'steps': steps}
