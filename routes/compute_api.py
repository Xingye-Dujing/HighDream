"""Synchronous/asynchronous compute, parse, plot and task-stream endpoints."""

from flask import Response, jsonify, request, stream_with_context
from flask.json import dumps as json_dumps

from routes.task_manager import task_manager
from utils.latex_formatter import str_to_latex

from .api_blueprint import api


@api.route('/parse', methods=['POST'])
def parse_input():
    """Parse a mathematical expression and convert it to LaTeX format.

    Expected JSON input:
        expression (str): The mathematical expression to parse
        operation_type (str): Type of operation for context.

    Returns:
        JSON response with either:
            success (bool): True if parsing succeeded
            result (str): LaTeX formatted output.
        or:
            success (bool): False if parsing failed
            error (str): Error message describing the issue.

    Example:
        POST /parse
        {
            "expression": "2+2*3",
            "operation_type": "arithmetic"
        }
    """
    data = request.json or {}
    expression = data.get('expression', '')
    operation_type = data.get('operation_type', '')

    try:
        latex_output = str_to_latex(expression, operation_type)
        return jsonify({'success': True, 'result': latex_output})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})


@api.route('/compute', methods=['POST'])
def compute():
    """Execute a mathematical computation based on operation type (async).

    Expected JSON input:
        operation_type (str): Type of operation to perform
        ... other fields depending on the operation type.

    Returns:
        JSON response with:
            success (bool): Always True for task creation
            task_id (str): Task ID for the polling result.
    """
    data = request.json or {}
    op_type = data.get('operation_type', '')

    task_id = task_manager.create_task(op_type, data)

    return jsonify({'success': True, 'task_id': task_id})


@api.route('/convert_form', methods=['POST'])
def convert_form():
    """Toggle a cell expression between LaTeX WYSIWYG and universal-box operator form.

    Expected JSON: {expression}. Returns {success, result} with the flipped form
    (unchanged if the expression is neither a recognisable operator nor LaTeX)."""
    data = request.json or {}
    from routes.op_form import toggle_form  # pylint: disable=import-outside-toplevel
    try:
        result = toggle_form(data.get('expression', ''))
        return jsonify({'success': True, 'result': result})
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return jsonify({'success': False, 'error': str(exc)})


@api.route('/steps', methods=['POST'])
def get_steps():
    """Return structured, per-step solver output used for step-linking (synchronous).

    Expected JSON input:
        operation_type (str): one of 'diff', 'integral', 'limit'
        expression (str): the mathematical expression
        variable (str): integration/differentiation variable
        plus point/direction (limit) or lower/upper (integral) as applicable.

    Returns:
        {success, op, steps: [{index, latex, reason, subtasks: [...]}]}
    """
    data = request.json or {}
    op_type = data.get('operation_type', '')
    if op_type not in ('diff', 'integral', 'limit'):
        return jsonify({'success': False, 'error': f'不支持的操作类型: {op_type}'})

    from core.step_tree import structured_solve  # pylint: disable=import-outside-toplevel
    return jsonify(structured_solve(op_type, data))


@api.route('/plot', methods=['POST'])
def plot_curve():
    """Sample a single-variable scalar function for client-side SVG plotting.

    Expected JSON input:
        expression (str), variable (str): the function y=f(x)
        xmin/xmax or lower/upper (number, optional): sample window (default +/-10)
        samples (int, optional): grid size (clamped)
        tangent_at (number, optional): also emit a tangent-line overlay

    Returns:
        JSON with success and points:[[x, y|null], ...] (null = asymptote/break),
        plus ymin/ymax, an optional tangent object and approximate real roots
        (``roots``: [{x, y}]) when the curve crosses / touches y=0; plain numbers only.
    """
    from utils.plot_data import sample_curve  # pylint: disable=import-outside-toplevel

    data = request.json or {}
    return jsonify(sample_curve(data))


@api.route('/stream')
def task_stream():
    """SSE endpoint that pushes task status updates in real time.

    The client opens an EventSource to ``/api/stream?task_id=...`` and receives:

    - ``status`` event while the task is pending or running
    - ``completed`` event with the full task dict on success
    - ``failed`` event on failure or if the task does not exist

    A ``heartbeat`` event is sent every 15 s to keep the connection alive.
    """
    task_id = request.args.get('task_id', '')

    def generate():
        last_status = None
        while True:
            task_info = task_manager.wait_for_task(task_id, timeout=3)
            if task_info is None:
                yield f"event: failed\ndata: {json_dumps({'error': '任务不存在: ' + task_id})}\n\n"
                return

            status = task_info['status']

            if status == 'completed':
                yield f"event: completed\ndata: {json_dumps(task_info)}\n\n"
                return
            if status == 'failed':
                yield f"event: failed\ndata: {json_dumps({'error': task_info.get('error', '')})}\n\n"
                return

            if status != last_status:
                last_status = status
                yield f"event: status\ndata: {json_dumps({'status': status})}\n\n"
            else:
                yield f"event: heartbeat\ndata: {{}}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
        }
    )


@api.route('/task_status', methods=['POST'])
def get_task_status():
    """Get the status and result of an async task.

    Expected JSON input:
        task_id (str): The task ID to query.

    Returns:
        JSON response with task status and results:
            task_id (str): Task ID
            operation_type (str): Operation type
            status (str): Task status (pending/running/completed/failed)
            result (various, optional): Computation result if completed
            error (str, optional): Error message if failed
            created_at (str): Task creation timestamp
            started_at (str, optional): Task start timestamp
            completed_at (str, optional): Task completion timestamp."""

    data = request.json or {}
    task_id = data.get('task_id', '')

    task_info = task_manager.get_task_status(task_id)
    if task_info:
        return jsonify({
            'success': True,
            'task': task_info
        })

    return jsonify({
        'success': False,
        'error': f'任务不存在: {task_id}'
    })
