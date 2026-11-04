"""Async method-tree enumeration endpoints."""

from flask import jsonify, request

from core.input_validation import as_bool_flag
from routes.task_manager import task_manager

from .api_blueprint import api


@api.route('/method_tree_start', methods=['POST'])
def method_tree_start():
    """Kick off an exhaustive rule-enumeration as an async task.

    Expected JSON input:
        domain (str): 'diff', 'integral', or 'limit'
        expression (str): Mathematical expression to enumerate
        variable (str, optional): Variable name (default 'x')
        point (str, optional): Limit point (default '0')
        direction (str, optional): Limit direction (default '+')
        max_depth (int, optional): Max root-to-leaf depth
        max_nodes (int, optional): Max total tree size
        time_limit_seconds (int, optional): Wall-clock budget

    Returns:
        JSON with success + task_id for polling via /api/method_tree_status.
    """
    data = request.json or {}
    if not data.get('domain'):
        return jsonify({'success': False, 'error': 'missing domain'})
    if not data.get('expression'):
        return jsonify({'success': False, 'error': 'missing expression'})

    interactive = as_bool_flag(data.get('interactive', False))
    if interactive:
        from config import METHOD_TREE_INTERACTIVE_TASK_TIMEOUT_SECONDS  # pylint: disable=import-outside-toplevel
        timeout = METHOD_TREE_INTERACTIVE_TASK_TIMEOUT_SECONDS
    else:
        from config import METHOD_TREE_TASK_TIMEOUT_SECONDS  # pylint: disable=import-outside-toplevel
        timeout = METHOD_TREE_TASK_TIMEOUT_SECONDS
    task_id = task_manager.create_task(
        'method_tree', data,
        timeout_seconds=timeout,
    )
    return jsonify({'success': True, 'task_id': task_id})


@api.route('/method_tree_status', methods=['POST'])
def method_tree_status():
    """Poll incremental progress or the final tree for a method-tree task.

    Expected JSON input:
        task_id (str): Task ID returned by /api/method_tree_start

    Returns:
        JSON with:
            success (bool)
            task (dict): standard task_manager status envelope
            tree (dict, optional): incremental snapshot while running, or
                the full final tree on completion.
    """
    from routes.method_tree_service import snapshot_method_tree  # pylint: disable=import-outside-toplevel

    data = request.json or {}
    task_id = data.get('task_id', '')

    task_info = task_manager.get_task_status(task_id)
    if task_info is None:
        return jsonify({
            'success': False,
            'error': f'任务不存在: {task_id}'
        })

    response = {'success': True, 'task': task_info}

    # While running, stream an incremental snapshot.
    if task_info['status'] == 'running':
        snapshot = snapshot_method_tree(task_id)
        if snapshot is not None:
            response['tree'] = snapshot
    elif task_info['status'] == 'completed':
        # The final tree lives in task.result.
        response['tree'] = task_info.get('result')

    return jsonify(response)


@api.route('/method_tree_cancel', methods=['POST'])
def method_tree_cancel():
    """Request cancellation of a running method-tree enumeration.

    Expected JSON input:
        task_id (str): Task ID to cancel

    Returns:
        JSON with success + cancelled (bool).
    """
    from routes.method_tree_service import cancel_method_tree  # pylint: disable=import-outside-toplevel

    data = request.json or {}
    task_id = data.get('task_id', '')
    cancelled = cancel_method_tree(task_id)
    return jsonify({'success': True, 'cancelled': cancelled})


@api.route('/method_tree_respond', methods=['POST'])
def method_tree_respond():
    """Respond to a pending interactive decision in a running enumeration.

    Expected JSON input:
        task_id (str): Task ID
        accepted (bool): True to apply the rule, False to skip

    Returns:
        JSON with success + responded (bool).
    """
    from routes.method_tree_service import respond_method_tree  # pylint: disable=import-outside-toplevel

    data = request.json or {}
    task_id = data.get('task_id', '')
    accepted = data.get('accepted', False)
    responded = respond_method_tree(task_id, bool(accepted))
    return jsonify({'success': True, 'responded': responded})


@api.route('/method_tree_path', methods=['POST'])
def method_tree_path():
    """Render one method-tree branch (a root->node rule path) as a linear derivation.

    Expected JSON input:
        domain, expression, variable, point, direction: the original problem
        rule_path (list[str]): rule names from the root to the chosen node
        node_id (str, optional): echoed back for the client's convenience

    Returns:
        JSON with success, result (a \\begin{align} block), steps, applied/requested, done.
    """
    from routes.method_tree_service import linear_path_method_tree  # pylint: disable=import-outside-toplevel

    data = request.json or {}
    return jsonify(linear_path_method_tree(data))
