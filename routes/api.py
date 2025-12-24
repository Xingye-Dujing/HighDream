"""API Routes Module for HighDream Application.

This module defines the Flask blueprint for all API endpoints, handling various
mathematical computation requests including expression parsing, matrix operations,
and process management.

Routes:
    /parse          - Parse mathematical expressions to LaTeX format
    /compute        - Execute mathematical computations (async)
    /task_status    - Get task status and results
    /matrix_analyze - Analyze matrix properties
    /matrix_cal     - Perform matrix calculations
    /get_process    - Retrieve process information
    /static/trees   - Serve static tree diagram files
"""

from sympy import Symbol, diff, integrate, latex, limit, simplify, sympify
from flask import Blueprint, request, jsonify, send_from_directory
from domains import MatrixAnalyzer, MatrixCalculator, ProcessManager
from config import TREES_DIR
from utils.latex_formatter import str_to_latex
from .task_manager import task_manager

# Create a Flask blueprint for API routes
api = Blueprint('api', __name__)


@api.route('/parse', methods=['POST'])
def parse_input():
    """Parse a mathematical expression and convert it to LaTeX format.

    Expected JSON input:
        expression (str): The mathematical expression to parse
        operation_type (str): Type of operation for context

    Returns:
        JSON response with either:
            success (bool): True if parsing succeeded
            result (str): LaTeX formatted output
        or:
            success (bool): False if parsing failed
            error (str): Error message describing the issue

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
        ... other fields depending on operation type

    Returns:
        JSON response with:
            success (bool): Always True for task creation
            task_id (str): Task ID for polling results
    """
    data = request.json or {}
    op_type = data.get('operation_type', '')

    # Create asynchronous task
    task_id = task_manager.create_task(op_type, data)

    return jsonify({
        'success': True,
        'task_id': task_id
    })


@api.route('/task_status', methods=['POST'])
def get_task_status():
    """Get the status and result of an async task.

    Expected JSON input:
        task_id (str): The task ID to query

    Returns:
        JSON response with task status and results:
            task_id (str): Task ID
            operation_type (str): Operation type
            status (str): Task status (pending/running/completed/failed)
            result (various, optional): Computation result if completed
            error (str, optional): Error message if failed
            created_at (str): Task creation timestamp
            started_at (str, optional): Task start timestamp
            completed_at (str, optional): Task completion timestamp
    """
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
    return jsonify(MatrixAnalyzer().analyze(expr))


@api.route('/matrix_cal', methods=['POST'])
def matrix_cal():
    """Perform step-by-step matrix calculations.

    Expected JSON input:
        expression (str): Initial matrix expression
        operations (str): Newline-separated list of operations to perform
        record_all_steps (str): "True" to record all calculation steps

    Returns:
        JSON response with:
            steps (list): List of LaTeX-formatted calculation steps
    """
    data = request.json or {}
    expr = data.get('expression', '')
    ops = data.get('operations', '').split('\n')
    record = data.get('record_all_steps') == 'True'

    calc = MatrixCalculator()
    calc.calculate(expr, ops, record)
    return jsonify({'steps': calc.get_steps_latex()})


@api.route('/get_process', methods=['POST'])
def get_process():
    """Retrieve information about a specific process.

    Expected JSON input:
        type (str): Type of process to retrieve
        expression (str): Expression related to the process

    Returns:
        JSON response containing process information from ProcessManager
    """
    data = request.json or {}
    ptype = data.get('type', '')
    expr = data.get('expression', '')
    return jsonify(ProcessManager.get_process(ptype, expr))


@api.route('/static/trees/<path:filename>')
def serve_tree(filename):
    """Serve static tree diagram files.

    Args:
        filename (str): Name of the file to serve from TREES_DIR

    Returns:
        File content from the trees directory
    """
    return send_from_directory(TREES_DIR, filename)


@api.route('/render_latex', methods=['POST'])
def render_latex():
    """Convert expression to LaTeX using sympy's latex function.

    Expected JSON input:
        expression (str): The mathematical expression to convert

    Returns:
        JSON response with either:
            success (bool): True if conversion succeeded
            latex (str): LaTeX formatted expression
        or:
            success (bool): False if conversion failed
            error (str): Error message describing the issue
    """
    data = request.json or {}
    expression = data.get('expression', '')

    try:
        latex_output = latex(sympify(expression))

        return jsonify({'success': True, 'latex': latex_output})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})


@api.route('/sympy_calculate', methods=['POST'])
def sympy_calculate():
    """Perform SymPy calculations based on operation type.

    Expected JSON input:
        expression (str): The mathematical expression to calculate
        operation_type (str): Type of operation ('diff', 'integral', 'limit')
        variable (str): Variable for the operation (default 'x')

    Returns:
        JSON response with either:
            success (bool): True if calculation succeeded
            result (str): Result of the SymPy calculation
        or:
            success (bool): False if calculation failed
            error (str): Error message describing the issue
    """
    data = request.json or {}
    expression = data.get('expression', '')
    operation_type = data.get('operation_type', 'diff')
    variable = data.get('variable', 'x')

    try:
        expr = sympify(expression)
        var = Symbol(variable)

        if operation_type == 'diff':
            result = diff(expr, var)
        elif operation_type == 'integral':
            result = integrate(expr, var)
        elif operation_type == 'limit':
            point = data.get('point', '0')
            direction = data.get('direction', '+-')
            point = sympify(point)
            result = limit(expr, var, point, dir=direction)
        else:
            # Default to just returning the parsed expression
            result = expr

        result_str = str(simplify(result))

        return jsonify({'success': True, 'result': result_str})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})


@api.route('/sympy_simplify', methods=['POST'])
def sympy_simplify():
    """Perform SymPy simplification on an expression.

    Expected JSON input:
        expression (str): The mathematical expression to simplify

    Returns:
        JSON response with either:
            success (bool): True if simplification succeeded
            result (str): Simplified result of the expression
        or:
            success (bool): False if simplification failed
            error (str): Error message describing the issue
    """
    data = request.json or {}
    expression = data.get('expression', '')

    try:
        expr = sympify(expression)
        result = simplify(expr)
        result_str = str(result)

        return jsonify({'success': True, 'result': result_str})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})
