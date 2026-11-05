"""Computation Service Module

This module provides a centralized interface for various mathematical computations
including calculus, linear algebra, and expression parsing operations.

The service routes requests to appropriate calculators based on operation type
and handles data processing, error handling, and result formatting.

Deployment note: the in-memory task/session tables and the local SQLite result
cache (utils/db_cache.py) only work on a single long-lived process (app_local.py).
On stateless platforms (e.g. Vercel) task/session state must be externalized and
HIGHDREAM_CACHE_DIR must point to a writable directory for the cache to work.
"""

import re
from typing import Any, Dict, Tuple

from sympy import Symbol, preorder_traversal, latex

from config import (DEFAULT_PARSER_MAX_DEPTH, DEFAULT_LHOPITAL_MAX_COUNT,
    HARD_MAX_DIFF_ORDER, HARD_MAX_PARSER_MAX_DEPTH, HARD_MAX_LHOPITAL_COUNT)
from core.input_validation import (
    as_bool_flag as _as_bool_flag,
    should_route_as_operator as _should_route_as_operator,
    normalize_matrix_input as _normalize_matrix_input,
    normalize_scalar_input as _normalize_scalar_input,
    user_error as _user_error,
)
from core.latex_syntax import (
    _extract_paren_arg, _find_innermost_op, _fold_matrix_ops,
    _iter_op_calls, _normalize_latex_ops, _split_top_args,
)
from core.matrix_ops import (
    matrix_charpoly as _matrix_charpoly,
    matrix_definiteness as _matrix_definiteness,
    matrix_jordan as _matrix_jordan,
    matrix_of as _matrix_of,
    matrix_pinv as _matrix_pinv,
    matrix_subspace as _matrix_subspace,
    matrix_trace as _matrix_trace,
)
from core.operator_registry import (
    _AUTO_DEFAULTS, _AUTO_FUNCS, _AUTO_MULTILINE_OPS, _AUTO_NAME_EXTRAS,
    _CALC_COMPOSABLE, _MATRIX_LATEX_OPS, _MATRIX_OP_NAMES,
    _SQUARE_MATRIX_OPS, _TEXT_INPUT_OPS,
)
from utils.cache_utils import cached_sympify
from utils.matrix_latex import latex_to_nested
from utils.assumptions import parse_assumption_flags
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


def _clamp_int(value, low: int, high: int) -> int:
    """Parse ``value`` as an int bounded to ``[low, high]``. Failures fall back to
    ``low`` so a huge/garbage order or depth cannot monopolize a worker."""
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        parsed = low
    return max(low, min(high, parsed))


def _derivative_var(obj):
    """The differentiation variable of a Derivative (handles (x,), (x, n), x)."""
    for spec in obj.args[1:]:
        if isinstance(spec, (tuple, list)):
            return spec[0]
        return spec
    return None


def _derivative_order(obj):
    """The order n of a Derivative: (x, n) -> n, else 1 (clamped)."""
    for spec in obj.args[1:]:
        if isinstance(spec, (tuple, list)) and len(spec) >= 2:
            return _clamp_int(spec[1], 1, HARD_MAX_DIFF_ORDER)
    return 1


def _diff_steps_latex(expr_str, var, order=1):
    """Differentiate ``expr_str`` ``order`` times, running the rule engine once per
    order and feeding the previous result back (never a black-box ``.diff(n)``), and
    return the combined step LaTeX plus the final value (or None if unresolved)."""
    order = _clamp_int(order, 1, HARD_MAX_DIFF_ORDER)
    calc = _calculators['diff']()
    cur = str(expr_str)
    exprs_all, notes_all = [], []
    for _k in range(order):
        steps, expl = calc.compute_list(cur, var)
        for step, note in zip(steps, expl):
            if step is not None:
                exprs_all.append(step)
                notes_all.append(note)
        concrete = [e for e in steps if e is not None]
        if not concrete:
            break
        cur = str(concrete[-1])
    concrete = [e for e in exprs_all if e is not None]
    value = concrete[-1] if concrete else None
    from core.base_step_generator import BaseStepGenerator  # pylint: disable=import-outside-toplevel
    gen = BaseStepGenerator()
    gen.steps = exprs_all
    gen._explanations = notes_all  # pylint: disable=protected-access
    return gen.to_latex(), value


def _dispatch_calculus(expression: str):
    """Auto-route a combined calculus expression by its top-level operator.

    d/dx -> differentiation, integral -> integration (with bounds), lim -> limit,
    sum -> series convergence, otherwise simplify the plain expression.
    """
    from sympy import Derivative, Integral, Limit, Product, Sum, simplify  # pylint: disable=import-outside-toplevel
    obj = cached_sympify(expression)

    if isinstance(obj, Derivative):
        var = _derivative_var(obj)
        if var is None:
            return False, '导数缺少求导变量'
        order = _derivative_order(obj)
        if order > 1:
            align, _val = _diff_steps_latex(str(obj.args[0]), var, order)
            return True, align
        return True, _calculators['diff']().compute_latex(str(obj.args[0]), var)

    if isinstance(obj, Integral):
        bounds = obj.args[1] if len(obj.args) > 1 else None
        var = bounds[0] if bounds else Symbol('x')
        kwargs = {}
        if bounds is not None and len(bounds) == 3:
            kwargs['lower'], kwargs['upper'] = bounds[1], bounds[2]
        # Universal-box integrals may carry parameters (e.g. a, n); honor the
        # integration variable and treat other symbols as constants (matching the
        # dedicated ``integral`` operation path).
        _calc = _calculators['integral']()
        _calc._allow_parameters = True
        return True, _calc.compute_latex(str(obj.args[0]), var, **kwargs)

    if isinstance(obj, Limit):
        direction = obj.args[3] if len(obj.args) > 3 else '+'
        direction = {'+-': 'both', 'pm': 'both'}.get(str(direction), str(direction))
        return True, LimitCalculator().compute_latex(str(obj.args[0]), obj.args[1], obj.args[2], direction)

    if isinstance(obj, Sum):
        from domains.applied import HANDLERS  # pylint: disable=import-outside-toplevel
        spec = obj.args[1] if len(obj.args) > 1 else (Symbol('n'),)
        d = {'operation_type': 'finite_sum', 'expression': str(obj.args[0]), 'variable': str(spec[0])}
        if len(spec) >= 3:  # bounded -> finite closed-form sum; else finite_sum falls back to convergence
            d['lower'], d['upper'] = str(spec[1]), str(spec[2])
        return HANDLERS['finite_sum'](d)

    if isinstance(obj, Product):
        from domains.applied import HANDLERS  # pylint: disable=import-outside-toplevel
        spec = obj.args[1] if len(obj.args) > 1 else (Symbol('n'),)
        d = {'operation_type': 'finite_product', 'expression': str(obj.args[0]), 'variable': str(spec[0])}
        if len(spec) >= 3:
            d['lower'], d['upper'] = str(spec[1]), str(spec[2])
        return HANDLERS['finite_product'](d)

    return True, r'\begin{align}' + latex(simplify(obj)) + r'\end{align}'


# ---------------------------------------------------------------------------
# Universal-box RECURSIVE compound evaluation.
#
# The plain ``auto`` dispatcher only recognized a single top-level operator, so
# ``3*diff(x,x)`` (an operator embedded in elementary arithmetic) fell through to
# a bare ``simplify`` -- no steps, and a violation of the "every computation shows
# its own steps" rule. ``_dispatch_compound`` instead finds each innermost
# ``name(args)`` operator call anywhere in the string, resolves it through the
# SAME per-op engines (reusing their step + value), splices the resulting value
# back into the surrounding expression, and repeats until only elementary
# arithmetic is left -- then renders one combined step-by-step derivation with
# each operator's own steps shown as a nested ``aligned`` sub-block.
# ---------------------------------------------------------------------------

_COMPOUND_MAX_STEPS = 64


def _build_op_data(name, args, user_data=None):
    """Assemble the per-op data payload from a function name + positional args
    (the same assembly the single-operator path uses, factored for reuse)."""
    op, fields = _AUTO_FUNCS[name]
    if name == 'limit' and len(args) == 3:
        # The universal box's \lim / limit(f,x,a) is two-sided; the dedicated
        # limit operation keeps its explicit UI direction (default right).
        args = list(args) + ['both']
    data = {'operation_type': op}
    if user_data:
        # An explicit operator template must not silently discard options the
        # caller set (type='2', target_form, use_cache, ...). Insert them before
        # the defaults so setdefault() cannot overwrite them, while keeping the
        # parsed operation and expression authoritative.
        for key, val in user_data.items():
            if key not in ('operation_type', 'expression', 'task_id'):
                data.setdefault(key, val)
    if name == 'parametric' and len(args) >= 2:
        # The palette's parametric(□,□) gives x(t) and y(t); the engine reads
        # them as two lines and differentiates both with respect to t.
        data['expression'] = args[0] + '\n' + args[1]
        data['variable'] = args[2] if len(args) >= 3 else 't'
        return data
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
    return data


def _render_inner_aligned(exprs, explanations):
    """Render (expr, note) engine steps as a nested ``aligned`` block (no literal $)."""
    from core.base_step_generator import format_note  # pylint: disable=import-outside-toplevel
    lines = []
    first = True
    for step, note in zip(exprs, explanations):
        if step is None:
            continue
        prefix = '&' if first else '&='
        first = False
        line = prefix + latex(step)
        if note:
            line += ' ' + format_note(note)
        lines.append(line)
    if not lines:
        return ''
    return r'\begin{aligned}' + r' \\ '.join(lines) + r'\end{aligned}'


def _as_aligned(latex_str):
    """Reuse an engine-produced ``\\begin{align}..\\end{align}`` as a nested aligned block."""
    if not isinstance(latex_str, str):
        return ''
    m = re.search(r'\\begin\{align\}(.*?)\\end\{align\}', latex_str, re.S)
    if not m:
        return ''
    return r'\begin{aligned}' + m.group(1).strip() + r'\end{aligned}'


def _auto_value(op, data):
    """Return (ok, steps_aligned_latex, value_expr_or_None) for one operator call.

    The engine produces both the shown steps and, for composable operators, a
    substitutable value. Non-composable operators (matrices / pedagogical objects)
    yield ``value=None`` so the caller can report it or (later phases) compose them."""
    from sympy import Derivative, Integral, Limit  # pylint: disable=import-outside-toplevel

    if op in ('diff', 'integral', 'limit'):
        from core.step_tree import _compute_steps  # pylint: disable=import-outside-toplevel
        if op == 'diff' and _clamp_int(data.get('order', '1'), 1, HARD_MAX_DIFF_ORDER) > 1:
            align, value = _diff_steps_latex(
                data.get('expression', ''), Symbol(str(data.get('variable', 'x') or 'x')),
                _clamp_int(data.get('order', '1'), 1, HARD_MAX_DIFF_ORDER))
            return True, _as_aligned(align), value
        try:
            exprs, explanations = _compute_steps(op, data)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, str(exc), None
        concrete = [e for e in exprs if e is not None]
        value = concrete[-1] if concrete else None
        if value is not None and hasattr(value, 'atoms') \
                and value.atoms(Derivative, Integral, Limit):
            value = None  # unresolved: never splice a half-solved operator
        steps = _render_inner_aligned(concrete, explanations)
        return True, steps, value

    if op == 'expr':
        expr = data.get('expression', '')
        try:
            value = cached_sympify(expr)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'无法解析表达式: {exc}', None
        return True, '', value

    # scalar-valued linear algebra: the result composes into elementary arithmetic.
    if op in ('det', 'rank', 'trace'):
        try:
            ok, res = _dispatch_compute(op, data)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'{op} 计算失败: {exc}', None
        if not ok:
            return False, res if isinstance(res, str) else str(res), None
        try:
            mat = _matrix_of(data.get('expression', ''))
            if op == 'det':
                value = mat.det()
            elif op == 'rank':
                value = mat.rank()
            else:
                value = mat.trace()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'矩阵解析失败: {exc}', None
        return True, _as_aligned(res), value

    # matrix-valued: value is a Matrix (not scalar-composable; guarded by the caller).
    if op in ('invert', 'ref', 'pinv'):
        try:
            ok, res = _dispatch_compute(op, data)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'{op} 计算失败: {exc}', None
        if not ok:
            return False, res if isinstance(res, str) else str(res), None
        try:
            from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
            mat = CommonMatrixCalculator.parse_matrix_input(data.get('expression', ''))
            if op == 'invert':
                value = mat.inv()
            elif op == 'pinv':
                value = mat.pinv()
            else:
                value = mat.rref()[0]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'矩阵运算失败: {exc}', None
        return True, _as_aligned(res), value

    # Not yet composable in this phase (S4 adds applied/series/complex scalar results).
    from domains.applied import HANDLERS  # pylint: disable=import-outside-toplevel
    if op in HANDLERS:
        try:
            ok, res = _dispatch_compute(op, data)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'{op} 计算失败: {exc}', None
        if not ok:
            return False, res if isinstance(res, str) else str(res), None
        # A handler that yields a composable scalar records it in ``_auto_value``
        # (single source of truth with its own shown steps); pedagogical results
        # (tangent equation, critical points, ODE family, ...) leave it unset.
        value = data.get('_auto_value')
        steps = _as_aligned(res) if isinstance(res, str) else ''
        return True, steps, value

    return True, '', None


def _dispatch_compound(original_display, text):
    """Recursively resolve every embedded operator call and render the combined
    step-by-step derivation. ``text`` is already latex-normalized + nested; the
    first align line shows ``original_display`` (exactly what the user typed)."""
    blocks = []
    seen = {text}
    guard = 0
    while True:
        guard += 1
        if guard > _COMPOUND_MAX_STEPS:
            return False, '表达式过于复杂或含无法消解的算子'
        loc = _find_innermost_op(text)
        if loc is None:
            break
        name = loc['name']
        op = _AUTO_FUNCS[name][0]
        args = _split_top_args(loc['inner'])
        if not args:
            return False, f'{name} 缺少参数'
        ok, steps, value = _auto_value(op, _build_op_data(name, args))
        if not ok:
            return False, steps
        if value is None:
            return False, f'{name}(...) 的结果目前不能作为子表达式代入外层运算'
        if hasattr(value, 'cols'):  # sympy Matrix result cannot enter scalar arithmetic
            return False, f'{name}(...) 的结果是矩阵,不能代入标量加减乘除;矩阵之间请用 add/subtract/multiply 运算'
        # splice the value as LaTeX so the working string stays uniformly LaTeX and
        # parse_latex handles it (str() would emit '**' that clashes with '^'/'\frac').
        text = text[:loc['start']] + r'\left(' + latex(value) + r'\right)' + text[loc['end']:]
        text = _normalize_latex_ops(text)
        if text in seen:
            return False, '计算未推进（可能存在循环定义）'
        seen.add(text)
        blocks.append((loc['raw'], steps, value))

    try:
        pre = cached_sympify(text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'初等合并失败: {exc}'
    final = pre.doit() if hasattr(pre, 'doit') else pre
    return True, _render_compound(original_display, blocks, pre, final)


def _render_compound(original_display, blocks, substituted, final_value):
    """One combined ``align``: original -> each operator's nested sub-steps ->
    the values substituted in -> final elementary merge."""
    lines = ['& ' + original_display]
    for call_raw, steps_latex, _value in blocks:
        if steps_latex:
            note = rf'\text{{算子 {call_raw} 的分步}}'
            lines.append(r'& \quad ' + note + r'\; ' + steps_latex)
    lines.append(r'&= ' + latex(substituted) + r' \quad \text{代入各算子结果}')
    lines.append(r'&= \boxed{' + latex(final_value) + '}')
    return r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


def _should_compound(text, calls):
    """Decide whether the universal box should run the recursive compound evaluator.

    Enters compound only when an operator sits in *elementary context* (arithmetic
    around it, several sibling calls) or a calculus operator has a genuinely nested
    operator inside. A single leading matrix/symbolic call is left to the fast path
    so its own engine still parses names like Inverse/Add."""
    if not calls:
        return False
    stripped = text.strip()
    leading = min(calls, key=lambda c: c['start'])
    indent = len(text) - len(text.lstrip())
    spans_whole = leading['start'] == indent and leading['raw'].strip() == stripped
    if not spans_whole:
        return True
    outer_op = _AUTO_FUNCS[leading['name']][0]
    if outer_op in _CALC_COMPOSABLE and _iter_op_calls(leading['inner']):
        return True
    return False


def _dispatch_auto(expression: str, user_data: Dict[str, Any] = None):
    """Universal single-box dispatcher (Maple-like): classify by operator/structure.

    Recognizes function-style operators (det/rank/integral/diff/taylor/...) for every
    supported operation, plus structural forms: d/dx, integral, limit, sum, matrix
    literals (A+B / A*B / lone matrix), partial (∂), and uppercase-symbol products
    (symbolic matrices). Delegates to the existing per-op dispatch so all step-by-step
    engines are reused."""
    from sympy import Derivative, Integral, Limit, Product, Sum  # pylint: disable=import-outside-toplevel

    # Any unfilled placeholder box -> one clear message, checked on the raw input
    # before conversion so latex_to_nested never trips over the empty slots.
    if re.search(r'\\square\b|\\Box\b|\\bigbox\b|\\placeholder\b', expression):
        return False, '请先填写输入框中的占位（□）后再计算'
    text = latex_to_nested(expression)  # bmatrix blocks -> [[...]] (no-op otherwise)

    # Fold matrix display spellings (M^{T}, M^{-1}, M^{+}, \det M, \mathrm{op} M,
    # M*M, M+M, M-M) into operator CALLS so the operator fast-path below dispatches the
    # requested matrix operation. Without this a lone "M^{T}" was treated as a bare
    # matrix and silently returned rref, dropping the operator entirely.
    text = _fold_matrix_ops(text)

    # 0) compound: an operator call embedded inside a larger elementary expression
    #    (3*diff(x,x), sin(x)+int(x^2,x), diff(int(x^2,x),x)) is resolved recursively
    #    by reusing each operator's own engine + steps, splicing values back. A lone
    #    top-level operator is left to the fast path below (byte-for-byte unchanged).
    norm = _normalize_latex_ops(text)
    calls = _iter_op_calls(norm)
    if _should_compound(norm, calls):
        return _dispatch_compound(expression, norm)

    # 1) function-style operator: name(args); bare \name<matrix> only for matrix ops
    scan = norm
    fm = re.match(r'\s*\\?([A-Za-z][A-Za-z0-9]*)\s*(\()?', scan)
    if fm:
        name = fm.group(1).lower()
        spec = _AUTO_FUNCS.get(name)
        bare_ok = name in _MATRIX_OP_NAMES
        if spec and (fm.group(2) or bare_ok):
            if fm.group(2):
                inner = _extract_paren_arg(scan, scan.index('(', fm.end() - 1))
                args = _split_top_args(inner) if inner is not None else []
            else:
                m2 = re.search(r'\[\[.*?\]\]', scan)
                args = [m2.group(0)] if m2 else []
            if args:
                op = spec[0]
                return _dispatch_compute(op, _build_op_data(name, args, user_data))

    # 2) matrix literals: arithmetic or lone matrix
    mats = re.findall(r'\[\[.*?\]\]', scan)
    if len(mats) >= 2:
        seg = scan[scan.index(mats[0]) + len(mats[0]):scan.index(mats[1])]
        op = 'subtract' if '-' in seg else ('multiply' if ('*' in seg or '\u00b7' in seg) else 'add')
        return _dispatch_compute('operations', {'operation_type': 'operations',
                                                'expression': mats[0] + '\n' + mats[1], 'operations': op})
    if len(mats) == 1:
        return _dispatch_compute('ref', {'operation_type': 'ref', 'expression': mats[0], 'target_form': 'rref'})

    # 3) structural operators (calculus) and simplify fallback
    try:
        obj = cached_sympify(scan)
    except Exception:  # pylint: disable=broad-exception-caught
        obj = None
    if obj is not None and isinstance(obj, (Derivative, Integral, Limit, Product, Sum)):
        return _dispatch_calculus(scan)
    return _dispatch_calculus(scan)


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
    if isinstance(value, dict):
        return {k: _strip_dollar(v) for k, v in value.items()}
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

        if not str(expression or '').strip():
            return False, '请输入表达式'

        # The symbolic-matrix palette key wraps the expression as msimplify(...);
        # the operation selector already says "symbolic matrix", so unwrap it.
        if operation_type == 'matrix_symbolic':
            wrapped = re.match(r'\s*(?:\\mathrm\{)?msimplify\}?\s*\((.*)\)\s*$',
                               str(expression), re.S)
            if wrapped:
                expression = wrapped.group(1)

        # The WYSIWYG palette inserts real operator templates (diff(...),
        # \int_0^1 ..., \det [matrix], M^{-1}, ...).  Honour that explicit intent
        # regardless of which operation selector is active; otherwise a user who
        # selected "求导" and then used the derivative key would differentiate a
        # Derivative and get the second derivative (a silent wrong answer).
        if _should_route_as_operator(expression, operation_type):
            return _dispatch_auto(expression, data)

        # Linear-algebra cells may carry MathLive LaTeX matrices; convert each
        # bmatrix/pmatrix block to the nested-list form the calculators parse.
        # No-op when the input is already plain text.
        if operation_type in _MATRIX_LATEX_OPS and operation_type != 'matrix_symbolic':
            ok, normalized = _normalize_matrix_input(
                expression, validate=operation_type != 'linear-system')
            if not ok:
                return False, normalized
            expression = normalized
        elif operation_type not in ('auto', 'calculus', 'matrix_symbolic') \
                and operation_type not in _TEXT_INPUT_OPS:
            ok, normalized = _normalize_scalar_input(
                expression,
                complex_mode=(operation_type.startswith('complex_')
                              or operation_type == 'de_moivre'),
            )
            if not ok:
                return False, normalized
            expression = normalized

        if operation_type in _SQUARE_MATRIX_OPS:
            from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
            try:
                matrix = CommonMatrixCalculator.parse_matrix_input(expression)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return False, f'矩阵无法解析: {exc}'
            if matrix.rows != matrix.cols:
                return False, f'该运算要求方阵，当前矩阵为 {matrix.rows}×{matrix.cols}'

        # Every downstream handler reads data['expression']; keep the canonical
        # normalized value there too (mutating this request's payload is safe and
        # lets applied handlers publish their composable _auto_value back).
        data['expression'] = expression

        use_cache = _as_bool_flag(data.get('use_cache', False))
        record_cache = _as_bool_flag(data.get('record_cache', False))

        # Optional user assumptions (e.g. "x>0, y real"). Applied as a final honest
        # simplify step on the calculators below. The result cache key (db_cache.
        # build_hash) carries no assumptions column, so any run WITH assumptions must
        # bypass the cache to avoid serving a derivation computed under other hypotheses.
        raw_assumptions = str(data.get('assumptions', '')).strip()
        assum_flags = parse_assumption_flags(raw_assumptions) if raw_assumptions else {}
        if assum_flags:
            use_cache = record_cache = False

        if operation_type == 'diff':
            order = _clamp_int(data.get('order', '1'), 1, HARD_MAX_DIFF_ORDER)
            if order > 1:  # higher derivative: run the engine once per order, feeding back
                align, _val = _diff_steps_latex(expression, Symbol(variable), order)
                return True, align
            if use_cache or record_cache:
                calc = _cached_calculators['diff']()
                calc.assum_flags = assum_flags
                calc.assumption_note = raw_assumptions
                return True, calc.compute_latex(expression, Symbol(variable),
                                                use_cache=use_cache, record_cache=record_cache)
            calc = _calculators['diff']()
            calc.assum_flags = assum_flags
            calc.assumption_note = raw_assumptions
            return True, calc.compute_latex(expression, Symbol(variable))

        if operation_type == 'integral':
            # UI sends the string 'True'/'False'; the universal box (_build_op_data)
            # omits the key, so a plain bool/absent value must still mean "special
            # matchers ON" (the documented default). str()-coercion keeps '' False-only
            # semantics without silently disabling quotient-diff / f*e^g rules.
            include_special_matchers = _as_bool_flag(
                data.get('include_special_matchers', True), default=True)
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
                calc.assum_flags = assum_flags
                calc.assumption_note = raw_assumptions
                calc._allow_parameters = True  # user path: allow parameters (a, n, ...)
                return True, calc.compute_latex(expression, Symbol(variable),
                                                use_cache=use_cache, record_cache=record_cache)
            calc = _calculators['integral'](include_special_matchers)
            calc.assum_flags = assum_flags
            calc.assumption_note = raw_assumptions
            calc._allow_parameters = True  # user path: allow parameters (a, n, ...)
            return True, calc.compute_latex(expression, Symbol(variable), **bounds)

        if operation_type == 'limit':
            point = cached_sympify(data.get('point', '0'))
            direction = data.get('direction', '+')
            max_lhopital = _clamp_int(
                data.get('max_lhopital_count', DEFAULT_LHOPITAL_MAX_COUNT),
                0, HARD_MAX_LHOPITAL_COUNT)
            if use_cache or record_cache:
                limiter = _cached_calculators['limit'](max_lhopital=max_lhopital)
                limiter.assum_flags = assum_flags
                limiter.assumption_note = raw_assumptions
                return True, limiter.compute_latex(expression, Symbol(variable), point, direction,
                                                   use_cache=use_cache, record_cache=record_cache)
            limiter = LimitCalculator(max_lhopital=max_lhopital)
            limiter.assum_flags = assum_flags
            limiter.assumption_note = raw_assumptions
            return True, limiter.compute_latex(expression, Symbol(variable), point, direction)

        if operation_type == 'calculus':
            return _dispatch_calculus(expression)

        if operation_type == 'auto':
            return _dispatch_auto(expression)

        if operation_type == 'ref':
            target = data.get('target_form', 'ref')
            return True, _calculators['ref']().compute_latex(expression, target)

        if operation_type == 'trace':
            return _matrix_trace(expression)

        if operation_type == 'pinv':
            return _matrix_pinv(expression)

        if operation_type == 'jordan':
            return _matrix_jordan(expression)

        if operation_type == 'definiteness':
            return _matrix_definiteness(expression)

        if operation_type == 'charpoly':
            return _matrix_charpoly(expression)

        if operation_type in ('nullspace', 'colspace', 'rowspace'):
            return _matrix_subspace(expression, operation_type)

        if operation_type == 'operations':
            ops = data.get('operations', '+')
            return True, _calculators['operations']().compute(expression, ops)

        if operation_type == 'invert':
            calc = _calculators['invert']()
            result = calc.auto_matrix_inverse(expression)
            if result is None:
                return False, '该矩阵不可逆（行列式为 0）'
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
            lines = [line for line in expression.split('\n') if line.strip()]
            if len(lines) < 2:
                return False, "向量投影需要两行：向量和基矩阵"
            calc = _calculators['projection']()
            result = calc.auto_project_vector(lines[0], lines[1])
            if result is None:
                return False, "向量投影失败：请检查向量与基矩阵的维度是否匹配"
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
            calc_type = data.get('type', 'basis_change')
            required = 3 if calc_type in ('coordinate_transform', 'coord_transform') else 2
            lines = [line for line in expression.split('\n') if line.strip()]
            if calc_type not in (
                'basis_change', 'change_basis', 'coordinate_transform', 'coord_transform',
                'standard_to_basis', 'std_to_basis', 'basis_to_standard', 'basis_to_std',
            ):
                return False, f'不支持的基变换类型: {calc_type}'
            if len(lines) < required:
                return False, f'该变换需要至少 {required} 行矩阵输入'
            calc = _calculators['transform_1']()
            calc.compute_transform('\n'.join(lines), calc_type)
            result = calc.get_steps_latex()
            if not result or result.replace(' ', '') == r'\begin{align}\end{align}':
                return False, '变换未产生结果，请检查输入的基/坐标矩阵'
            return True, result

        if operation_type == 'transform-2':
            calc_type = data.get('type', 'find_matrix')
            required = 3 if calc_type in (
                'change_basis', 'basis_change', 'change_transform_basis') else 2
            if calc_type not in (
                'find_matrix', 'find_transform_matrix',
                'change_basis', 'basis_change', 'change_transform_basis',
            ):
                return False, f'不支持的线性变换类型: {calc_type}'
            lines = [line for line in expression.split('\n') if line.strip()]
            if len(lines) < required:
                return False, f'该变换需要至少 {required} 行矩阵输入'
            calc = _calculators['transform_2']()
            calc.compute('\n'.join(lines), calc_type)
            result = calc.get_steps_latex()
            if not result or result.replace(' ', '') == r'\begin{align}\end{align}':
                return False, '变换未产生结果，请检查输入的变换/基矩阵'
            return True, result

        if operation_type == 'expr':
            max_depth = _clamp_int(
                data.get('max_depth', DEFAULT_PARSER_MAX_DEPTH),
                1, HARD_MAX_PARSER_MAX_DEPTH)
            sort_strategy = data.get('sort_strategy', 'complexity')

            def length_sort(expr):
                """Sort expressions by string length."""
                return len(str(expr))

            def complexity_sort(expr):
                """Sort expressions by complexity (node count)."""
                # Simple complexity estimate: node counting
                return sum(1 for _ in preorder_traversal(expr))

            strategy = complexity_sort
            if sort_strategy == 'length':
                strategy = length_sort

            parser = ExpressionParser(max_depth=max_depth, sort_strategy=strategy)

            # Parse returns list of (expr, reason) tuples
            results = parser.parse(expression)
            expressions_out = []
            for expr_obj, reason in results:
                expr_latex = latex(expr_obj)
                expressions_out.append({'latex': expr_latex, 'reason': str(reason)})

            if not expressions_out:
                # Nothing found (or the strategy yielded none): still show a result
                # instead of an empty "未找到等价表达式".
                try:
                    simplified = cached_sympify(expression).simplify()
                except Exception:  # pylint: disable=broad-exception-caught
                    simplified = expression
                expressions_out.append({'latex': latex(simplified), 'reason': '化简'})

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
            return True, r"\begin{align}" + r" \\ ".join(lines) + r"\end{align}"

        # pylint: disable=import-outside-toplevel
        from domains.applied import HANDLERS
        handler = HANDLERS.get(operation_type)
        if handler is not None:
            try:
                return handler(data)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                return False, _user_error(exc, operation_type)

        return False, f"不支持的操作类型: {operation_type}"

    except Exception as e:  # pylint: disable=broad-exception-caught
        return False, _user_error(e, operation_type)
