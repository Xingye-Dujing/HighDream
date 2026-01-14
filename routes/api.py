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

import traceback
import sympy as sp
from sympy import (
    # Basic classes
    Symbol, symbols, sympify, Expr, Integer, Rational, Float, Piecewise,
    # Algebra
    simplify, expand, factor, collect, apart, together, cancel,
    # Calculus
    diff, integrate, limit, Derivative, Integral, Limit,
    # Equation solving
    solve, solveset, linsolve, nonlinsolve, rsolve,
    # Matrices
    Matrix, ImmutableMatrix, eye, zeros, ones, diag, randMatrix,
    # Linear algebra
    det, transpose,
    # Trigonometric functions
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    # Exponential and logarithmic
    exp, log,
    # Other functions
    sqrt, Abs, sign, conjugate, re, im, arg,
    # Constants
    pi, E, oo, I, zoo, nan,
    # Summations, products
    Sum, Product, summation, product,
    # Series
    series, limit_seq,
    # Combinatorics
    binomial, factorial, gamma,
    # Polynomials
    Poly,
    # Differential equations
    Function, dsolve, Eq,
    # Vectors
    MatrixSymbol,
    # LaTeX output
    latex, pretty,
    # Numerical calculation
    N, nsimplify,
)
from sympy.simplify.fu import fu
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
    expression = simplify(data.get('expression', ''))
    operation_type = data.get('operation_type', 'diff')
    variable = data.get('variable', 'x')

    try:
        expr = sympify(expression)
        var = Symbol(variable, real=True)

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

        result_str = str(simplify(fu(result)))

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
        result = simplify(sympify(expression))
        result_str = str(result)

        return jsonify({'success': True, 'result': result_str})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})


@api.route('/sympy_execute', methods=['POST'])
def sympy_execute():
    """Execute arbitrary SymPy code and return the result.

    This endpoint allows users to write and execute SymPy code directly,
    supporting functions like diff, integrate, limit, simplify, etc.

    Expected JSON input:
        code (str): The SymPy code to execute

    Returns:
        JSON response with either:
            success (bool): True if execution succeeded
            result (str): Result of the SymPy code execution
            latex (str): LaTeX formatted result (if applicable)
        or:
            success (bool): False if execution failed
            error (str): Error message describing the issue
    """
    data = request.json or {}
    code = data.get('code', '')
    x, y, t, a, b = symbols('x y t a b', real=True)

    if not code:
        return jsonify({'success': False, 'error': 'Code cannot be empty'})

    try:
        # Prepare execution environment with all SymPy functions
        exec_globals = {
            # SymPy modules
            'sp': sp,
            'sympy': sp,
            # Basic classes and functions
            'Symbol': Symbol,
            'symbols': symbols,
            'sympify': sympify,
            'Expr': Expr,
            'Integer': Integer,
            'Rational': Rational,
            'Float': Float,
            'Piecewise': Piecewise,
            # Algebra
            'simplify': simplify,
            'expand': expand,
            'factor': factor,
            'collect': collect,
            'apart': apart,
            'together': together,
            'cancel': cancel,
            # Calculus
            'diff': diff,
            'integrate': integrate,
            'limit': limit,
            'Derivative': Derivative,
            'Integral': Integral,
            'Limit': Limit,
            # Equation solving
            'solve': solve,
            'solveset': solveset,
            'linsolve': linsolve,
            'nonlinsolve': nonlinsolve,
            'dsolve': dsolve,
            'rsolve': rsolve,
            'Eq': Eq,
            # Matrices
            'Matrix': Matrix,
            'ImmutableMatrix': ImmutableMatrix,
            'eye': eye,
            'zeros': zeros,
            'ones': ones,
            'diag': diag,
            'randMatrix': randMatrix,
            # Linear algebra
            'det': det,
            'transpose': transpose,
            # Trigonometric functions
            'sin': sin,
            'cos': cos,
            'tan': tan,
            'cot': cot,
            'sec': sec,
            'csc': csc,
            'asin': asin,
            'acos': acos,
            'atan': atan,
            'acot': acot,
            'asec': asec,
            'acsc': acsc,
            'sinh': sinh,
            'cosh': cosh,
            'tanh': tanh,
            'coth': coth,
            'sech': sech,
            'csch': csch,
            'asinh': asinh,
            'acosh': acosh,
            'atanh': atanh,
            'acoth': acoth,
            'asech': asech,
            'acsch': acsch,
            # Exponential and logarithmic
            'exp': exp,
            'log': log,
            'ln': log,
            # Other functions
            'sqrt': sqrt,
            'Abs': Abs,
            'sign': sign,
            'conjugate': conjugate,
            're': re,
            'im': im,
            'arg': arg,
            # Constants
            'pi': pi,
            'E': E,
            'oo': oo,
            'I': I,
            'zoo': zoo,
            'nan': nan,
            # Summations, products
            'Sum': Sum,
            'Product': Product,
            'summation': summation,
            'product': product,
            # Series
            'series': series,
            'limit_seq': limit_seq,
            # Combinatorics
            'binomial': binomial,
            'factorial': factorial,
            'gamma': gamma,
            # Polynomials
            'Poly': Poly,
            # Function class
            'Function': Function,
            # Vectors
            'MatrixSymbol': MatrixSymbol,
            # LaTeX output
            'latex': latex,
            'pretty': pretty,
            # Numerical calculation
            'N': N,
            'nsimplify': nsimplify,
            # Compatibility
            'print': print,
            'range': range,
            'len': len,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            'sum': sum,
            # Symbols
            'x': x,
            'y': y,
            't': t,
            'a': a,
            'b': b,
        }

        # Execute code
        exec_locals = {}
        exec(code, exec_globals, exec_locals)

        # Get last expression as result
        result = None
        for key, value in exec_locals.items():
            if not key.startswith('_'):
                result = value

        # If no result, try to extract the last expression from code
        if result is None:
            # Try to parse the last expression in the code
            lines = code.strip().split('\n')
            last_line = lines[-1].strip()
            if last_line and not last_line.startswith('#'):
                try:
                    result = eval(last_line, exec_globals, exec_locals)
                except Exception:
                    pass

        if result is None:
            return jsonify({'success': True, 'result': 'Code executed successfully but no return value'})

        # Auto simplify
        if isinstance(result, Expr):
            result = simplify(result)

        # Try to convert to string
        result_str = str(result)

        # Try to convert to LaTeX
        try:
            result_latex = latex(result)
        except Exception:
            result_latex = None

        return jsonify({
            'success': True,
            'result': result_str,
            'latex': result_latex
        })

    except Exception as e:
        error_msg = f'Execution error: {str(e)}'
        # Add detailed error information
        error_details = traceback.format_exc()
        return jsonify({'success': False, 'error': error_msg, 'details': error_details})
