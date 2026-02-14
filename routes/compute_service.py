"""Computation Service Module

This module provides a centralized interface for various mathematical computations
including calculus, linear algebra, and expression parsing operations.

The service routes requests to appropriate calculators based on operation type
and handles data processing, error handling, and result formatting.
"""

# import uuid
# form config import TREES_DIR
from typing import Any, Dict, Tuple

from sympy import Symbol, preorder_traversal, latex, sympify

from config import DEFAULT_PARSER_MAX_DEPTH, DEFAULT_LHOPITAL_MAX_COUNT
from domains import (
    DiffCalculator, IntegralCalculator, LimitCalculator, ExpressionParser,
    RefCalculator, BasicOperations, Inverter, Rank, LUDecomposition,
    Diagonalization, SVDSolver, SchurDecomposition, EigenSolver,
    DeterminantCalculator, DetCalculator, OrthogonalProcessor,
    VectorProjectionSolver, LinearDependence, LinearTransform,
    LinearSystemConverter, LinearSystemSolver, BaseTransform
)

# Registry of available calculators mapped by operation type
_calculators = {
    'diff': DiffCalculator,
    'expr': ExpressionParser,
    'integral': IntegralCalculator,
    'ref': RefCalculator,
    'operations': BasicOperations,
    'invert': Inverter,
    'rank': Rank,
    'LU': LUDecomposition,
    'diag': Diagonalization,
    'svd': SVDSolver,
    'eigen': EigenSolver,
    'schur': SchurDecomposition,
    'det_1': DeterminantCalculator,
    'det_2': DetCalculator,
    'orthogonal': OrthogonalProcessor,
    'projection': VectorProjectionSolver,
    'dependence': LinearDependence,
    'transform_1': BaseTransform,
    'transform_2': LinearTransform,
    'linear_system_converter': LinearSystemConverter,
    'linear_solver': LinearSystemSolver,
}


def start_compute(operation_type: str, data: Dict[str, Any]) -> Tuple[bool, Any]:
    """Execute mathematical computation based on operation type and input data.

    This function serves as the main entry point for all computational operations,
    routing requests to appropriate calculators and handling result formatting.

    Args:
        operation_type (str): Type of mathematical operation to perform
        data (Dict[str, Any]): Input parameters for the computation.

    Returns:
        Tuple[bool, Any]: A tuple containing:
            - success flag (bool)
            - result data or error message (Any)

    Supported Operations:
        - diff: Differentiation
        - integral: Integration
        - limit: Limit calculation
        - ref: Row echelon form
        - operations: Basic matrix operations
        - invert: Matrix inversion
        - rank: Matrix rank calculation
        - LU: LU decomposition
        - diag: Matrix diagonalization
        - svd: Singular value decomposition
        - eigen: Eigenvalue/eigenvector calculation
        - schur: Schur decomposition
        - det: Determinant calculation
        - orthogonal: Orthogonality analysis
        - projection: Vector projection
        - dependence: Linear dependence check
        - linear-system: Linear system conversion
        - linear-solver: Linear system solving
        - transform-1: Basis transformation
        - transform-2: Linear transformation
        - expr: Expression parsing and analysis
    """
    try:
        expression = data.get('expression', '')
        variable = data.get('variable', 'x')

        if operation_type == 'diff':
            return True, _calculators['diff']().compute_latex(expression, Symbol(variable))  # type: ignore

        if operation_type == 'integral':
            include_special_matchers = data.get('include_special_matchers', True) == 'True'
            return (True, _calculators['integral'](include_special_matchers).
                    compute_latex(expression, Symbol(variable)))  # type: ignore

        if operation_type == 'limit':
            point = sympify(data.get('point', '0'))
            direction = data.get('direction', '+')
            max_lhopital = int(data.get('max_lhopital_count', DEFAULT_LHOPITAL_MAX_COUNT))
            limiter = LimitCalculator(max_lhopital=max_lhopital)
            return True, limiter.compute_latex(expression, Symbol(variable), point, direction)

        if operation_type == 'ref':
            target = data.get('target_form', 'ref')
            return True, _calculators['ref']().compute_latex(expression, target)

        if operation_type == 'operations':
            ops = data.get('operations', '+')
            return True, _calculators['operations']().compute(expression, ops)

        if operation_type == 'invert':
            calc = _calculators['invert']()
            calc.auto_matrix_inverse(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'rank':
            calc = _calculators['rank']()
            calc.auto_matrix_rank(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'LU':
            calc = _calculators['LU']()
            calc.auto_lu_decomposition(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'diag':
            calc = _calculators['diag']()
            calc.auto_diagonalization(expression)
            return True, calc.get_steps_latex()

        if operation_type in ('svd', 'singular'):
            calc = _calculators['svd']()
            if operation_type == 'svd':
                calc.compute_svd(expression)
            else:
                calc.compute_singular_values_only(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'eigen':
            calc = _calculators['eigen']()
            calc.auto_eigen_solver(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'schur':
            calc = _calculators['schur']()
            calc.auto_schur_decomposition(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'det':
            cal_type = data.get('type', '1')
            calc = _calculators['det_1'](
            ) if cal_type == '1' else _calculators['det_2']()
            return True, calc.compute_latex(expression)

        if operation_type == 'orthogonal':
            calc = _calculators['orthogonal']()
            calc.auto_orthogonal_analysis(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'projection':
            lines = expression.split('\n')
            if len(lines) < 2:
                return False, "Requires two lines: vector and basis"
            calc = _calculators['projection']()
            calc.auto_project_vector(lines[0], lines[1])
            return True, calc.get_steps_latex()

        if operation_type == 'dependence':
            calc = _calculators['dependence']()
            calc.auto_check_dependence(expression)
            return True, calc.get_steps_latex()

        if operation_type == 'linear-system':
            conv = _calculators['linear_system_converter']()
            exprs, unknowns = conv.str_to_Eq(expression, get_unknowns=True)
            conv.show_equations_to_matrix(exprs, unknowns)
            return True, conv.get_steps_latex()

        if operation_type == 'linear-solver':
            parts = expression.split('\n')
            if len(parts) < 2:
                return False, "The first line should be matrix A, second line vector b."
            A, b = parts[0], parts[1]
            solver = _calculators['linear_solver']()
            solver.solve(A, b)
            return True, solver.get_steps_latex()

        if operation_type == 'transform-1':
            calc = _calculators['transform_1']()
            calc_type = data.get('type', 'basis_change')
            calc.compute_transform(expression, calc_type)
            return True, calc.get_steps_latex()

        if operation_type == 'transform-2':
            calc = _calculators['transform_2']()
            calc_type = data.get('type', 'find_matrix')
            calc.compute(expression, calc_type)
            return True, calc.get_steps_latex()

        if operation_type == 'expr':
            max_depth = int(data.get('max_depth', DEFAULT_PARSER_MAX_DEPTH))
            # is_draw_tree = data.get('is_draw_tree') == 'true'
            sort_strategy = data.get('sort_strategy', 'complexity')

            def length_sort(expr):
                """Sort expressions by string length."""
                return len(str(expr))

            def complexity_sort(expr):
                """Sort expressions by complexity (node count)."""
                # Simple complexity estimate: node counting
                return sum(1 for _ in preorder_traversal(expr))

            strategy = None
            if sort_strategy == 'length':
                strategy = length_sort
            elif sort_strategy == 'complexity':
                strategy = complexity_sort

            parser = ExpressionParser(max_depth=max_depth, sort_strategy=strategy)

            # Parse returns list of (expr, reason) tuples
            results = parser.parse(expression)
            expressions_out = []
            for expr_obj, reason in results:
                expr_latex = latex(expr_obj)
                expressions_out.append({'latex': expr_latex, 'reason': str(reason)})

            # # Generate derivation tree SVG and return accessible static path
            # tree_svg_url = None
            # if is_draw_tree:
            #     tree = parser.parse_tree(expression)
            #     # Generate unique filename
            #     file_name = f"{uuid.uuid4().hex}.svg"
            #     save_path = f"{TREES_DIR}/{file_name}"

            #     parser.draw_expr_tree(tree, save_path=save_path)
            #     tree_svg_url = save_path

            # return True, (expressions_out, tree_svg_url)
            return True, (expressions_out,)

        return False, f"Unsupported operation type: {operation_type}"

    except Exception as e:
        return False, str(e)
