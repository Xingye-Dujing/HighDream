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
from utils.matrix_latex import latex_to_nested
from domains import (
    CachedDiffCalculator, CachedIntegralCalculator, CachedLimitCalculator,
    DiffCalculator, IntegralCalculator, LimitCalculator, ExpressionParser,
    RefCalculator, BasicOperations, Inverter, Rank, LUDecomposition,
    Diagonalization, SVDSolver, SchurDecomposition, EigenSolver,
    DeterminantCalculator, DetCalculator, OrthogonalProcessor,
    VectorProjectionSolver, LinearDependence, LinearTransform,
    LinearSystemConverter, LinearSystemSolver, BaseTransform
)

# Operation types that accept MathLive LaTeX matrices (bmatrix/pmatrix). The
# converter is a no-op for plain-text input, so all linear-algebra ops are listed.
_MATRIX_LATEX_OPS = {
    'ref', 'operations', 'invert', 'rank', 'LU', 'diag', 'svd', 'singular',
    'eigen', 'schur', 'det', 'orthogonal', 'projection', 'dependence',
    'linear-system', 'linear-solver', 'transform-1', 'transform-2', 'matrix_symbolic',
}

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


def _derivative_var(obj):
    """The differentiation variable of a Derivative (handles (x,), (x, n), x)."""
    for spec in obj.args[1:]:
        if isinstance(spec, (tuple, list)):
            return spec[0]
        return spec
    return None


def _dispatch_calculus(expression: str):
    """Auto-route a combined calculus expression by its top-level operator.

    d/dx -> differentiation, integral -> integration (with bounds), lim -> limit,
    sum -> series convergence, otherwise simplify the plain expression.
    """
    from sympy import Derivative, Integral, Limit, Sum, simplify  # pylint: disable=import-outside-toplevel
    obj = cached_sympify(expression)

    if isinstance(obj, Derivative):
        var = _derivative_var(obj)
        if var is None:
            return False, '导数缺少求导变量'
        return True, _calculators['diff']().compute_latex(str(obj.args[0]), var)

    if isinstance(obj, Integral):
        bounds = obj.args[1] if len(obj.args) > 1 else None
        var = bounds[0] if bounds else Symbol('x')
        kwargs = {}
        if bounds is not None and len(bounds) == 3:
            kwargs['lower'], kwargs['upper'] = bounds[1], bounds[2]
        return True, _calculators['integral']().compute_latex(str(obj.args[0]), var, **kwargs)

    if isinstance(obj, Limit):
        direction = obj.args[3] if len(obj.args) > 3 else '+'
        return True, LimitCalculator().compute_latex(str(obj.args[0]), obj.args[1], obj.args[2], direction)

    if isinstance(obj, Sum):
        from domains.applied import HANDLERS  # pylint: disable=import-outside-toplevel
        return HANDLERS['series_convergence']({'expression': str(obj.args[0]), 'variable': str(obj.args[1][0])})

    return True, r'\begin{align}' + latex(simplify(obj)) + r'\end{align}'


def _extract_paren_arg(text: str, open_idx: int):
    """Return the substring inside the balanced parentheses starting at open_idx."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return text[open_idx + 1:i]
    return None


def _split_top_args(argstr: str):
    """Split a function argument string on top-level commas (paren/bracket aware)."""
    args, depth, cur = [], 0, ''
    for ch in argstr:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth -= 1
        if ch == ',' and depth == 0:
            args.append(cur.strip())
            cur = ''
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


# name -> (operation_type, ordered data fields). Args fill the fields positionally;
# 'variable'/'direction'/'type'/'operations'/'target_form' fall back to defaults.
_AUTO_FUNCS = {
    # calculus
    'diff': ('diff', ['expression', 'variable']), 'derivative': ('diff', ['expression', 'variable']),
    'd': ('diff', ['expression', 'variable']),
    'int': ('integral', ['expression', 'variable', 'lower', 'upper']),
    'integrate': ('integral', ['expression', 'variable', 'lower', 'upper']),
    'limit': ('limit', ['expression', 'variable', 'point']),
    'sum': ('series_convergence', ['expression', 'variable']),
    'series': ('series_convergence', ['expression', 'variable']),
    'converge': ('series_convergence', ['expression', 'variable']),
    'taylor': ('taylor', ['expression', 'variable', 'point', 'order']),
    'maclaurin': ('taylor', ['expression', 'variable', 'point', 'order']),
    'taylorerror': ('taylor_error', ['expression', 'variable', 'point', 'order']),
    'radius': ('radius_of_convergence', ['expression', 'variable']),
    'radiusofconvergence': ('radius_of_convergence', ['expression', 'variable']),
    'powerseries': ('power_series', ['expression', 'variable', 'point']),
    'improper': ('improper_integral', ['expression', 'variable', 'lower', 'upper']),
    'improperintegral': ('improper_integral', ['expression', 'variable', 'lower', 'upper']),
    'iterated': ('iterated_integral', ['expression', 'variable', 'variable2', 'lower', 'upper', 'lower2', 'upper2']),
    'doubleintegral': ('iterated_integral', ['expression', 'variable', 'variable2', 'lower', 'upper', 'lower2', 'upper2']),
    # applications of differentiation / integration
    'tangent': ('tangent', ['expression', 'variable', 'point']),
    'linearization': ('linearization', ['expression', 'variable', 'point']),
    'linearize': ('linearization', ['expression', 'variable', 'point']),
    'critical': ('critical_points', ['expression', 'variable']),
    'extrema': ('critical_points', ['expression', 'variable']),
    'newton': ('newton', ['expression', 'variable', 'point', 'order']),
    'meanvalue': ('mean_value', ['expression', 'variable', 'lower', 'upper']),
    'rolle': ('mean_value', ['expression', 'variable', 'lower', 'upper']),
    'arclength': ('arc_length', ['expression', 'variable', 'lower', 'upper']),
    'volume': ('volume_disk', ['expression', 'variable', 'lower', 'upper']),
    'disk': ('volume_disk', ['expression', 'variable', 'lower', 'upper']),
    'shell': ('volume_shell', ['expression', 'variable', 'lower', 'upper']),
    'surface': ('surface_area', ['expression', 'variable', 'lower', 'upper']),
    'implicit': ('implicit_diff', ['expression', 'variable']),
    'parametric': ('parametric_diff', ['expression', 'variable']),
    'polararea': ('polar_area', ['expression', 'variable', 'lower', 'upper']),
    'polarslope': ('polar_slope', ['expression', 'variable']),
    'ode': ('ode', ['expression', 'variable']),
    'dsolve': ('ode', ['expression', 'variable']),
    'numerical': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    'numericalintegration': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    'simpson': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    'trapezoid': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    # complex
    'complex': ('complex_simplify', ['expression']),
    'complexsimplify': ('complex_simplify', ['expression']),
    'complexpolar': ('complex_polar', ['expression']),
    'complexroots': ('complex_roots', ['expression', 'variable']),
    'demoivre': ('de_moivre', ['expression', 'order']),
    # linear algebra
    'det': ('det', ['expression']), 'determinant': ('det', ['expression']),
    'rank': ('rank', ['expression']),
    'inverse': ('invert', ['expression']), 'inv': ('invert', ['expression']),
    'rref': ('ref', ['expression']), 'ref': ('ref', ['expression']), 'rowreduce': ('ref', ['expression']),
    'eigen': ('eigen', ['expression']), 'eigenvals': ('eigen', ['expression']),
    'eigenvects': ('eigen', ['expression']),
    'svd': ('svd', ['expression']), 'singular': ('svd', ['expression']),
    'schur': ('schur', ['expression']),
    'lu': ('LU', ['expression']),
    'diag': ('diag', ['expression']), 'diagonalize': ('diag', ['expression']),
    'orthogonal': ('orthogonal', ['expression']), 'orthogonalize': ('orthogonal', ['expression']),
    'gramschmidt': ('orthogonal', ['expression']),
    'dependent': ('dependence', ['expression']), 'dependence': ('dependence', ['expression']),
    'project': ('projection', ['expression']), 'projection': ('projection', ['expression']),
    # symbolic / expression
    'simplify': ('expr', ['expression']), 'expand': ('expr', ['expression']),
    'factor': ('expr', ['expression']), 'equivalent': ('expr', ['expression']),
    'msimplify': ('matrix_symbolic', ['expression']),
    'matrixsimplify': ('matrix_symbolic', ['expression']),
    'symmatrix': ('matrix_symbolic', ['expression']),
    # matrix arithmetic via the universal box (two matrix args joined by newline)
    'add': ('operations', []), 'subtract': ('operations', []), 'minus': ('operations', []),
    'multiply': ('operations', []), 'matmul': ('operations', []),
}

# op -> fixed extra data fields the calculators require.
_AUTO_DEFAULTS = {
    'det': {'type': '1'}, 'ref': {'target_form': 'rref'}, 'svd': {},
    'expr': {'sort_strategy': 'simplify'},
}

# Ops whose calculators read a single ``expression`` holding several matrices /
# vectors separated by newlines: every function argument is joined with '\n'.
_AUTO_MULTILINE_OPS = {
    'operations', 'projection', 'dependence', 'orthogonal',
}

# Function name -> fixed data fields that the name itself decides.
_AUTO_NAME_EXTRAS = {
    'add': {'operations': '+'}, 'subtract': {'operations': '-'}, 'minus': {'operations': '-'},
    'multiply': {'operations': '*'}, 'matmul': {'operations': '*'},
}

# Matrix-function names that may appear bare (e.g. \det [[1,2],[3,4]]).
_MATRIX_OP_NAMES = {
    'det', 'determinant', 'rank', 'inverse', 'inv', 'rref', 'ref', 'rowreduce',
    'eigen', 'eigenvals', 'eigenvects', 'svd', 'singular', 'schur', 'lu', 'diag',
    'diagonalize', 'orthogonal', 'orthogonalize', 'gramschmidt', 'dependent',
    'dependence', 'project', 'projection', 'transpose',
}


def _dispatch_auto(expression: str):
    """Universal single-box dispatcher (Maple-like): classify by operator/structure.

    Recognizes function-style operators (det/rank/integral/diff/taylor/...) for every
    supported operation, plus structural forms: d/dx, integral, limit, sum, matrix
    literals (A+B / A*B / lone matrix), partial (∂), and uppercase-symbol products
    (symbolic matrices). Delegates to the existing per-op dispatch so all step-by-step
    engines are reused."""
    import re  # pylint: disable=import-outside-toplevel
    from sympy import Derivative, Integral, Limit, Sum  # pylint: disable=import-outside-toplevel
    text = latex_to_nested(expression)  # bmatrix blocks -> [[...]] (no-op otherwise)

    # Any unfilled placeholder box -> one clear message, before routing.
    if re.search(r'\\square\b|\\Box\b|\\bigbox\b', text):
        return False, '请先填写输入框中的占位（□）后再计算'

    # 1) function-style operator: name(args); bare \name<matrix> only for matrix ops
    fm = re.match(r'\s*\\?([A-Za-z]+)\s*(\()?', text)
    if fm:
        name = fm.group(1).lower()
        spec = _AUTO_FUNCS.get(name)
        bare_ok = name in _MATRIX_OP_NAMES
        if spec and (fm.group(2) or bare_ok):
            if fm.group(2):
                inner = _extract_paren_arg(text, text.index('(', fm.end() - 1))
                args = _split_top_args(inner) if inner is not None else []
            else:
                m2 = re.search(r'\[\[.*?\]\]', text)
                args = [m2.group(0)] if m2 else []
            if args:
                op, fields = spec
                data = {'operation_type': op}
                if op in _AUTO_MULTILINE_OPS:
                    data['expression'] = '\n'.join(args)
                else:
                    for i, field in enumerate(fields):
                        if i < len(args):
                            data[field] = args[i]
                for key, val in _AUTO_NAME_EXTRAS.get(name, {}).items():
                    data.setdefault(key, val)
                for key, val in _AUTO_DEFAULTS.get(op, {}).items():
                    data.setdefault(key, val)
                return _dispatch_compute(op, data)

    # 2) matrix literals: arithmetic or lone matrix
    mats = re.findall(r'\[\[.*?\]\]', text)
    if len(mats) >= 2:
        seg = text[text.index(mats[0]) + len(mats[0]):text.index(mats[1])]
        op = 'subtract' if '-' in seg else ('multiply' if ('*' in seg or '\u00b7' in seg) else 'add')
        return _dispatch_compute('operations', {'operation_type': 'operations',
                                                'expression': mats[0] + '\n' + mats[1], 'operations': op})
    if len(mats) == 1:
        return _dispatch_compute('ref', {'operation_type': 'ref', 'expression': mats[0], 'target_form': 'rref'})

    # 3) structural operators (calculus) and simplify fallback
    try:
        obj = cached_sympify(text)
    except Exception:  # pylint: disable=broad-exception-caught
        obj = None
    if obj is not None and isinstance(obj, (Derivative, Integral, Limit, Sum)):
        return _dispatch_calculus(text)
    return _dispatch_calculus(text)


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

        # Linear-algebra cells may carry MathLive LaTeX matrices; convert each
        # bmatrix/pmatrix block to the nested-list form the calculators parse.
        # No-op when the input is already plain text.
        if operation_type in _MATRIX_LATEX_OPS:
            expression = latex_to_nested(expression)

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

        if operation_type == 'calculus':
            return _dispatch_calculus(expression)

        if operation_type == 'auto':
            return _dispatch_auto(expression)

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
        from domains.applied import HANDLERS
        handler = HANDLERS.get(operation_type)
        if handler is not None:
            try:
                return handler(data)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return False, f'计算失败: {exc}'

        return False, f"Unsupported operation type: {operation_type}"

    except Exception as e:
        return False, str(e)
