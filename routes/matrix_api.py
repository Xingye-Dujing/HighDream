"""Matrix analyze / calculate / process endpoints."""

from flask import jsonify, request

from domains import MatrixAnalyzer, MatrixCalculator, ProcessManager

from .api_blueprint import api


@api.route('/matrix_analyze', methods=['POST'])
def matrix_analyze():
    """Analyze properties of a matrix expression.

    Expected JSON input:
        expression (str): Matrix expression to analyze

    Returns:
        JSON response containing analysis results from MatrixAnalyzer
    """
    data = request.json or {}
    expr = data.get('expression', '')
    from utils.matrix_latex import latex_to_nested  # pylint: disable=import-outside-toplevel
    try:
        expr = latex_to_nested(expr)
        return jsonify(MatrixAnalyzer().analyze(expr))
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc)})


@api.route('/matrix_cal', methods=['POST'])
def matrix_cal():
    """Perform step-by-step matrix calculations.

    Expected JSON input:
        expression (str): Initial matrix expression
        operations (str): Newline-separated list of operations to perform
        record_all_steps (str): "True" to record all calculation steps.

    Returns:
        JSON response with:
            steps (list): List of LaTeX-formatted calculation steps.
    """
    data = request.json or {}
    expr = data.get('expression', '')
    ops = [o for o in data.get('operations', '').split('\n') if o.strip()]
    record = data.get('record_all_steps') == 'True'
    if not ops:
        return jsonify({'success': False, 'error': '请输入矩阵操作', 'steps': ''})
    from utils.matrix_latex import latex_to_nested  # pylint: disable=import-outside-toplevel
    try:
        expr = latex_to_nested(expr)
        ops = [latex_to_nested(o) for o in ops]
    except ValueError as exc:
        return jsonify({'success': False, 'error': str(exc), 'steps': ''})

    calc = MatrixCalculator()
    try:
        calc.calculate(expr, ops, record)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        from routes.compute_service import _user_error  # pylint: disable=import-outside-toplevel
        return jsonify({'success': False, 'error': _user_error(exc, '矩阵操作'), 'steps': ''})
    return jsonify({'success': True, 'steps': calc.get_steps_latex()})


@api.route('/get_process', methods=['POST'])
def get_process():
    """Retrieve information about a specific process.

    Expected JSON input:
        type (str): Type of process to retrieve
        expression (str): Expression related to the process.

    Returns:
        JSON response containing process information from ProcessManager
    """
    data = request.json or {}
    ptype = data.get('type', '')
    expr = data.get('expression', '')
    from utils.matrix_latex import latex_to_nested  # pylint: disable=import-outside-toplevel
    try:
        expr = latex_to_nested(expr)
        return jsonify(ProcessManager.get_process(ptype, expr))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return jsonify({'success': False, 'error': str(exc), 'process': ''})
