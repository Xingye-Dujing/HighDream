"""Matrix analyze / calculate / process endpoints."""

from flask import jsonify, request

from core.input_validation import as_bool_flag
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
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    expr = data.get('expression', '')
    if not isinstance(expr, str) or not expr.strip():
        return jsonify({'success': False, 'error': '请输入矩阵表达式'})
    from utils.matrix_latex import latex_to_nested  # pylint: disable=import-outside-toplevel
    try:
        expr = latex_to_nested(expr)
        return jsonify(MatrixAnalyzer().analyze(expr))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Any SymPy failure inside analyze() must become a JSON error, not an
        # HTML 500 that breaks the client's res.json() call.
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
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    from utils.matrix_latex import latex_to_nested  # pylint: disable=import-outside-toplevel
    try:
        expr = data.get('expression', '')
        operations = data.get('operations', '')
        if not isinstance(expr, str) or not isinstance(operations, str):
            return jsonify({'success': False, 'error': '请输入矩阵表达式和矩阵操作', 'steps': ''})
        ops = [o for o in operations.split('\n') if o.strip()]
        if not ops:
            return jsonify({'success': False, 'error': '请输入矩阵操作', 'steps': ''})
        record = as_bool_flag(data.get('record_all_steps'))
        expr = latex_to_nested(expr)
        ops = [latex_to_nested(o) for o in ops]
    except (ValueError, TypeError, AttributeError) as exc:
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
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
    ptype = data.get('type', '')
    expr = data.get('expression', '')
    from utils.matrix_latex import latex_to_nested  # pylint: disable=import-outside-toplevel
    try:
        expr = latex_to_nested(expr)
        return jsonify(ProcessManager.get_process(ptype, expr))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return jsonify({'success': False, 'error': str(exc), 'process': ''})
