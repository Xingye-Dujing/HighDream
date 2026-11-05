"""Manual rule-selection session endpoints (state is intentionally in-memory)."""

import threading
import time
import uuid

from flask import jsonify, request

from core.manual_step_solver import ManualStepSolver

from .api_blueprint import api

# Sessions are tiny but never freed by the client unless the page calls
# manual_reset; cap them so a long-lived server cannot grow without bound.
_MANUAL_SESSION_MAX = 200
_MANUAL_SESSION_TTL_SECONDS = 3600
_manual_sessions: dict = {}
# One re-entrant lock guards the session dict *and* every solver operation.
# Because an in-flight step holds this same lock, _prune_manual_sessions (which
# also needs it) can never run mid-step, so a session still in use is never
# evicted. RLock lets manual_start call the pruner while already holding it.
_manual_sessions_lock = threading.RLock()


def _prune_manual_sessions() -> None:
    """Drop expired sessions, then the oldest ones beyond the cap.

    Must be called while holding ``_manual_sessions_lock`` (directly or
    re-entrantly). That mutual exclusion is what protects in-use sessions.
    """
    now = time.monotonic()
    with _manual_sessions_lock:
        expired = [sid for sid, (_, created) in _manual_sessions.items()
                   if now - created > _MANUAL_SESSION_TTL_SECONDS]
        for sid in expired:
            _manual_sessions.pop(sid, None)
        overflow = len(_manual_sessions) - _MANUAL_SESSION_MAX
        if overflow > 0:
            oldest = sorted(_manual_sessions.items(), key=lambda kv: kv[1][1])[:overflow]
            for sid, _ in oldest:
                _manual_sessions.pop(sid, None)


def _get_manual_solver(session_id: str):
    """Return a session's solver, or None. Caller must hold the session lock."""
    entry = _manual_sessions.get(session_id)
    return entry[0] if entry else None


def _manual_action(session_id, method_name, *args):
    """Run one solver method and return the endpoint's JSON response.

    The lock is held across fetching the solver, mutating it and serializing
    the resulting state, so no two requests can mutate the same solver at
    once and pruning cannot evict a session while it is being stepped.
    """
    with _manual_sessions_lock:
        try:
            solver = _get_manual_solver(session_id)
            if solver is None:
                return jsonify({'success': False, 'error': 'Session not found'})
            state = getattr(solver, method_name)(*args)
            return jsonify({'success': True, 'state': state})
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return jsonify({'success': False, 'error': str(exc)})


@api.route('/manual_start', methods=['POST'])
def manual_start():
    """Create a new manual step-by-step solver session.

    Expected JSON input:
        domain (str): 'diff', 'integral', or 'limit'
        expression (str): Mathematical expression to compute
        variable (str, optional): Variable name (default 'x')
        point (str, optional): Limit point (default '0')
        direction (str, optional): Limit direction (default '+')

    Returns:
        JSON with:
            success (bool): True if session created
            session_id (str): Session ID
            state (dict): Initial solver state
    """
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    domain = data.get('domain', '')
    expression = data.get('expression', '')
    variable = data.get('variable', 'x')
    point = data.get('point', '0')
    direction = data.get('direction', '+')

    try:
        solver = ManualStepSolver(
            domain=domain, expression=expression,
            variable=variable, point=point, direction=direction,
        )
        with _manual_sessions_lock:
            _prune_manual_sessions()
            session_id = uuid.uuid4().hex
            _manual_sessions[session_id] = (solver, time.monotonic())
            state = solver.state()
        return jsonify({
            'success': True,
            'session_id': session_id,
            'state': state,
        })
    except Exception as e:  # pylint: disable=broad-exception-caught
        return jsonify({'success': False, 'error': str(e)})


@api.route('/manual_step', methods=['POST'])
def manual_step():
    """Apply a selected rule to the current expression.

    Expected JSON input:
        session_id (str): Session ID
        rule_name (str): Name of the rule to apply

    Returns:
        JSON with:
            success (bool): True if step succeeded
            state (dict): Updated solver state
    """
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    return _manual_action(
        data.get('session_id', ''), 'apply_rule', data.get('rule_name', '')
    )


@api.route('/manual_undo', methods=['POST'])
def manual_undo():
    """Undo the last applied rule / SymPy fallback, restoring the pre-step state.

    Expected JSON input:
        session_id (str): Session ID

    Returns:
        JSON with:
            success (bool): True if a step was undone
            state (dict): Restored solver state (same shape as /manual_step)
    """
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    return _manual_action(data.get('session_id', ''), 'undo')


@api.route('/manual_fallback', methods=['POST'])
def manual_fallback():
    """Apply SymPy's .doit() fallback for the current expression.

    Expected JSON input:
        session_id (str): Session ID

    Returns:
        JSON with:
            success (bool): True if fallback succeeded
            state (dict): Updated solver state
    """
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    return _manual_action(data.get('session_id', ''), 'fallback')


@api.route('/manual_reset', methods=['POST'])
def manual_reset():
    """Delete a manual solver session.

    Expected JSON input:
        session_id (str): Session ID to delete

    Returns:
        JSON with:
            success (bool): True if session was deleted
    """
    data = request.json or {}
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    session_id = data.get('session_id', '')
    with _manual_sessions_lock:
        try:
            _manual_sessions.pop(session_id, None)
        except TypeError:
            # Unhashable id can never match a real session; reset is idempotent.
            pass
    return jsonify({'success': True})
