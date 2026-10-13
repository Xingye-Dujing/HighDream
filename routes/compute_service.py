"""Computation Service Module

This module provides a centralized interface for various mathematical computations
including calculus, linear algebra, and expression parsing operations.

The service routes requests to appropriate calculators based on operation type
and handles data processing, error handling, and result formatting.
"""

from typing import Any, Dict, Tuple

from sympy import Symbol, preorder_traversal, latex

from config import DEFAULT_PARSER_MAX_DEPTH, DEFAULT_LHOPITAL_MAX_COUNT
from utils.cache_utils import cached_sympify
from domains import (
    CachedDiffCalculator, CachedIntegralCalculator, CachedLimitCalculator,
    DiffCalculator, IntegralCalculator, LimitCalculator, ExpressionParser,
    RefCalculator, BasicOperations, Inverter, Rank, LUDecomposition,
    Diagonalization, SVDSolver, SchurDecomposition, EigenSolver,
    DeterminantCalculator, DetCalculator, OrthogonalProcessor,
    VectorProjectionSolver, LinearDependence, LinearTransform,
    LinearSystemConverter, LinearSystemSolver, BaseTransform
)

# Cached calculator variants used when use_cache or record_cache is set
_cached_calculators = {
    'diff': CachedDiffCalculator,
    'integral': CachedIntegralCalculator,
    'limit': CachedLimitCalculator,
}

# Class lookup keyed by operation name. NOTE: the authoritative routing is the
# if/elif chain in start_compute(), not this dict -- the runtime operation_type
# strings it compares (e.g. 'det', 'transform-1', 'linear-system') differ from
# several dict keys here ('det_1' / 'transform_1' / 'linear_system_converter').
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
    """Dispatch a computation and guarantee its string result contains no literal ``$``.

    All step output is expected to be live LaTeX (math in math mode, prose in
    ``\\text{}``); any stray ``$`` delimiter that slipped through is stripped so
    nothing renders as raw text. Non-string results (e.g. the ``expr`` tuple) are
    returned unchanged.
    """
    ok, result = _dispatch_compute(operation_type, data)
    return ok, _strip_dollar(result) if ok else result


def _strip_dollar(value):
    """Recursively remove literal ``$`` from string results (lists/tuples/dicts ok)."""
    if isinstance(value, str):
        return value.replace('$', '')
    if isinstance(value, list):
        return [_strip_dollar(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_strip_dollar(v) for v in value)
    return value


def _dispatch_compute(operation_type: str, data: Dict[str, Any]) -> Tuple[bool, Any]:
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
        - method_tree: Exhaustive rule-enumeration tree (see
          core.method_tree_enumerator)
    """
    try:
        if operation_type == 'method_tree':
            # Imported lazily so the rest of compute_service stays light.
            from routes.method_tree_service import start_method_tree  # pylint: disable=import-outside-toplevel
            return start_method_tree(data)

        expression = data.get('expression', '')
        variable = data.get('variable', 'x')

        use_cache = data.get('use_cache', False) == 'True'
        record_cache = data.get('record_cache', False) == 'True'

        if operation_type == 'diff':
            if use_cache or record_cache:
                calc = _cached_calculators['diff']()
                return True, calc.compute_latex(expression, Symbol(variable),
                                                use_cache=use_cache, record_cache=record_cache)
            return True, _calculators['diff']().compute_latex(expression, Symbol(variable))

        if operation_type == 'integral':
            include_special_matchers = data.get('include_special_matchers', True) == 'True'
            bounds = {}
            lower_raw = str(data.get('lower', '')).strip()
            upper_raw = str(data.get('upper', '')).strip()
            if upper_raw != '':
                bounds['upper'] = cached_sympify(upper_raw)
                bounds['lower'] = cached_sympify(lower_raw) if lower_raw != '' else 0
            # Result cache is keyed only by op/expr/var/point/direction (db_cache.
            # build_hash) and has no columns for bounds, so it is used only for
            # indefinite integrals; definite integrals bypass it to avoid collisions.
            if (use_cache or record_cache) and not bounds:
                calc = _cached_calculators['integral'](include_special_matchers)
                return True, calc.compute_latex(expression, Symbol(variable),
                                                use_cache=use_cache, record_cache=record_cache)
            return True, _calculators['integral'](include_special_matchers). \
                compute_latex(expression, Symbol(variable), **bounds)

        if operation_type == 'limit':
            point = cached_sympify(data.get('point', '0'))
            direction = data.get('direction', '+')
            max_lhopital = int(data.get('max_lhopital_count', DEFAULT_LHOPITAL_MAX_COUNT))
            if use_cache or record_cache:
                limiter = _cached_calculators['limit'](max_lhopital=max_lhopital)
                return True, limiter.compute_latex(expression, Symbol(variable), point, direction,
                                                   use_cache=use_cache, record_cache=record_cache)
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

            # NOTE: derivation-tree rendering (parse_tree/draw_expr_tree) was removed
            # as unused; the result stays a 1-tuple so the frontend reads result[0].
            return True, (expressions_out,)

        if operation_type == 'matrix_symbolic':
            # pylint: disable=import-outside-toplevel
            from domains.matrix.matrix_symbolic import simplify_matrix_symbolic
            try:
                steps = simplify_matrix_symbolic(data.get('expression', ''))
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return False, f'符号矩阵化简失败: {exc}'
            lines = []
            for idx, (mexpr, reason) in enumerate(steps):
                head = '&' if idx == 0 else '&='
                line = head + latex(mexpr)
                if reason:
                    line += rf" \quad \text{{{reason}}}"
                lines.append(line)
            return True, r"\begin{align}" + r"\newline".join(lines) + r"\end{align}"

        # pylint: disable=import-outside-toplevel
        from core.applied_calc import HANDLERS
        handler = HANDLERS.get(operation_type)
        if handler is not None:
            try:
                return handler(data)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return False, f'计算失败: {exc}'

        return False, f"Unsupported operation type: {operation_type}"

    except Exception as e:
        return False, str(e)
