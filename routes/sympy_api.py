"""Raw SymPy / LaTeX helper endpoints used by the word editor and tools."""

import traceback

import sympy as sp
from flask import jsonify, request
from sympy import (
    Abs, Derivative, Eq, Expr, Float, Function, ImmutableMatrix, Integer,
    Integral, Limit, Matrix, MatrixSymbol, N, Piecewise, Poly, Product,
    Rational, Sum, Symbol, binomial, collect, conjugate, cos, cosh, cot,
    coth, csc, csch, diag, diff, dsolve, eye, factor, factorial, gamma,
    im, integrate, limit, limit_seq, log, nan, nsimplify, oo, ones, pi,
    pretty, product, randMatrix, re, rsolve, sec, sech, series, sign, sin,
    sinh, solve, solveset, sqrt, summation, symbols, sympify, tan, tanh,
    exp, arg,
    transpose, zeros, det, apart, together, cancel, expand, linsolve, nonlinsolve,
    asin, acos, atan, acot, asec, acsc, asinh, acosh, atanh, acoth,
    asech, acsch, latex, I, E, zoo, simplify,
)
from sympy.simplify.fu import fu

from utils.cache_utils import cached_sympify

from .api_blueprint import api


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
            error (str): Error message describing the issue.
    """
    data = request.json or {}
    expression = data.get('expression', '')

    try:
        latex_output = latex(cached_sympify(expression))

        return jsonify({'success': True, 'latex': latex_output})
    except Exception as e:
        return jsonify({'success': False, 'error': f'{e}'})


@api.route('/sympy_calculate', methods=['POST'])
def sympy_calculate():
    """Perform SymPy calculations based on the operation type.

    Expected JSON input:
        expression (str): The mathematical expression to calculate
        operation_type (str): Type of operation ('diff', 'integral', 'limit')
        variable (str): Variable for the operation (default 'x')

    Returns:
        JSON response with either:
            success (bool): True if calculation succeeded
            the result (str): Result of the SymPy calculation
        or:
            success (bool): False if calculation failed
            error (str): Error message describing the issue.
    """
    data = request.json or {}
    expression = data.get('expression', '')
    operation_type = data.get('operation_type', 'diff')
    variable = data.get('variable', 'x')

    try:
        expr = cached_sympify(expression)
        # No assumptions on the working symbol: the expression's own symbols are
        # assumption-free (sympify), and sympy treats Symbol('x', real=True) as a
        # *different* symbol, which made diff/integral/limit silently return
        # 0 / x*x / x for every input.
        var = Symbol(variable)

        if operation_type == 'diff':
            result = diff(expr, var)
        elif operation_type == 'integral':
            result = integrate(expr, var)
        elif operation_type == 'limit':
            point = data.get('point', '0')
            direction = data.get('direction', '+-')
            point = cached_sympify(str(point))
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
            error (str): Error message describing the issue.
    """
    data = request.json or {}
    expression = data.get('expression', '')

    try:
        result = simplify(cached_sympify(expression))
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
            error (str): Error message describing the issue.
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

        # Get last expression as the result
        result = None
        for key, value in exec_locals.items():
            if not key.startswith('_'):
                result = value

        # If no result, try to extract the last expression from code.
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
