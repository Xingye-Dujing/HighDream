"""Computation Service Module

This module provides a centralized interface for various mathematical computations
including calculus, linear algebra, and expression parsing operations.

The service routes requests to appropriate calculators based on operation type
and handles data processing, error handling, and result formatting.
"""

import re
from typing import Any, Dict, Tuple

from sympy import Symbol, preorder_traversal, latex, sympify

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
    'trace', 'nullspace', 'colspace', 'rowspace',
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


def _derivative_order(obj):
    """The order n of a Derivative: (x, n) -> n, else 1."""
    for spec in obj.args[1:]:
        if isinstance(spec, (tuple, list)) and len(spec) >= 2:
            try:
                return max(1, int(spec[1]))
            except (TypeError, ValueError):
                return 1
    return 1


def _diff_steps_latex(expr_str, var, order=1):
    """Differentiate ``expr_str`` ``order`` times, running the rule engine once per
    order and feeding the previous result back (never a black-box ``.diff(n)``), and
    return the combined step LaTeX plus the final value (or None if unresolved)."""
    order = max(1, int(order or 1))
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


def _matrix_of(expression):
    from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
    return CommonMatrixCalculator.parse_matrix_input(expression)


def _matrix_trace(expression):
    """tr(A) = sum of the main diagonal, shown as its own step."""
    try:
        mat = _matrix_of(expression)
        total = mat.trace()
        diag = [mat[i, i] for i in range(min(mat.rows, mat.cols))]
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'矩阵解析失败: {exc}'
    diag_latex = ' + '.join(latex(d) for d in diag) if diag else '0'
    lines = ['& ' + latex(mat),
             r'&= ' + diag_latex + r' \quad \text{对角线元素之和}',
             r'&= \boxed{' + latex(total) + r'}']
    return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


def _matrix_subspace(expression, kind):
    """Row/column/null space via reduced row echelon form, shown step by step."""
    try:
        mat = _matrix_of(expression)
        rref_mat, pivots = mat.rref()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'矩阵解析失败: {exc}'
    title = {'nullspace': '零空间 N(A)', 'colspace': '列空间 C(A)', 'rowspace': '行空间 R(A)'}[kind]
    pivot_latex = ', '.join(str(p + 1) for p in pivots) if pivots else '无'
    lines = ['& ' + latex(mat),
             r'&= ' + latex(rref_mat) + r' \quad \text{行最简形, 主元在第 } ' + pivot_latex + r' \text{ 列}']
    if kind == 'nullspace':
        basis = mat.nullspace()
    elif kind == 'rowspace':
        basis = [rref_mat.row(i).T for i in range(rref_mat.rows)
                 if any(not rref_mat[i, j].is_zero for j in range(rref_mat.cols))]
    else:  # colspace: pivot columns of the ORIGINAL matrix
        basis = [mat.col(p) for p in pivots]
    for vec in basis:
        lines.append(r'&\rightarrow\ ' + latex(vec))
    lines.append(r'&\quad \text{' + title + ' 的一组基如上' +
                 (f', 维数 = {len(basis)}' if basis else ', 维数 = 0') + '}')
    return True, r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


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
        return True, _calculators['integral']().compute_latex(str(obj.args[0]), var, **kwargs)

    if isinstance(obj, Limit):
        direction = obj.args[3] if len(obj.args) > 3 else '+'
        return True, LimitCalculator().compute_latex(str(obj.args[0]), obj.args[1], obj.args[2], direction)

    if isinstance(obj, Sum):
        from domains.applied import HANDLERS  # pylint: disable=import-outside-toplevel
        spec = obj.args[1] if len(obj.args) > 1 else (Symbol('n'),)
        d = {'operation_type': 'finite_sum', 'expression': str(obj.args[0]), 'variable': str(spec[0])}
        if len(spec) >= 3:  # bounded -> finite closed-form sum; else finite_sum falls back to convergence
            d['lower'], d['upper'] = str(spec[1]), str(spec[2])
        return HANDLERS['finite_sum'](d)

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
    'diff': ('diff', ['expression', 'variable', 'order']),
    'derivative': ('diff', ['expression', 'variable', 'order']),
    'd': ('diff', ['expression', 'variable', 'order']),
    'int': ('integral', ['expression', 'variable', 'lower', 'upper']),
    'integrate': ('integral', ['expression', 'variable', 'lower', 'upper']),
    'limit': ('limit', ['expression', 'variable', 'point']),
    'sum': ('finite_sum', ['expression', 'variable', 'lower', 'upper']),
    'fsum': ('finite_sum', ['expression', 'variable', 'lower', 'upper']),
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
    'trace': ('trace', ['expression']),
    'nullspace': ('nullspace', ['expression']), 'nullity': ('nullspace', ['expression']),
    'colspace': ('colspace', ['expression']), 'columnspace': ('colspace', ['expression']),
    'rowspace': ('rowspace', ['expression']),
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
    'trace', 'nullspace', 'nullity', 'colspace', 'columnspace', 'rowspace',
}


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
_OP_CALL_RX = re.compile(r'(?<![A-Za-z_\\])([A-Za-z]+)\s*\(')


def _paren_close_index(text, open_idx):
    """Index of the ')' matching the '(' at ``open_idx`` (balanced), else None."""
    depth = 0
    for i in range(open_idx, len(text)):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i
    return None


def _grab_balanced_paren(text, search_from):
    """If a balanced ``(...)`` group starts at/after ``search_from`` (only spaces
    skipped), return ``(inner, end_idx_after_close)``; else ``(None, search_from)``."""
    i = search_from
    while i < len(text) and text[i] == ' ':
        i += 1
    if i >= len(text) or text[i] != '(':
        return None, search_from
    close = _paren_close_index(text, i)
    if close is None:
        return None, search_from
    return text[i + 1:close], close + 1


def _iter_op_calls(text):
    """All ``name(...)`` calls whose ``name`` is a supported operator, in order.

    Each entry: name/start/open/end/inner/raw. Unknown names (sin, exp, foo) are
    intentionally NOT treated as operators so they flow through as plain functions.
    """
    calls = []
    for match in _OP_CALL_RX.finditer(text):
        name = match.group(1).lower()
        if name not in _AUTO_FUNCS:
            continue
        open_idx = match.end() - 1
        close = _paren_close_index(text, open_idx)
        if close is None:
            continue
        calls.append({'name': name, 'start': match.start(), 'open': open_idx,
                      'end': close + 1, 'inner': text[open_idx + 1:close],
                      'raw': text[match.start():close + 1]})
    return calls


def _find_innermost_op(text):
    """The first operator call whose arguments contain no further operator call.

    Innermost-first resolution guarantees each engine call receives fully
    resolved, operator-free arguments."""
    calls = _iter_op_calls(text)
    for call in calls:
        if not _iter_op_calls(call['inner']):
            return call
    return calls[0] if calls else None


def _normalize_latex_ops(text):
    """Rewrite the LaTeX operator forms the MathLive keyboard emits into the
    functional ``name(args)`` form the compound scanner understands. Each operator's
    parenthesized body (and an integral's trailing ``dX``) is CONSUMED so it is not
    duplicated. Unrecognized spellings are left untouched (handled in a later phase).
    Bodies are normalized recursively so nested operators convert too."""
    return _nl_limit(_nl_integral(_nl_partial(_nl_diff(_nl_sum(text)))))


def _nl_sum(src):
    """``\\sum_{var=low}^{high} (BODY)`` -> ``sum(BODY, var, low, high)``."""
    out, i = [], 0
    pat = re.compile(r'\\sum\s*_\{([^{}]*?)\}(?:\s*\^\{([^{}]*)\})?')
    while True:
        match = pat.search(src, i)
        if not match:
            out.append(src[i:])
            break
        sub = match.group(1)
        high = (match.group(2) or '').strip()
        mvar = re.match(r'\s*(\w+)\s*=\s*(.+)', sub)
        var = mvar.group(1) if mvar else (sub.strip() or 'n')
        low = mvar.group(2).strip() if mvar else ''
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        nb = _normalize_latex_ops(body)
        if high != '':
            out.append(rf'sum({nb},{var},{low or 1},{high})')
        else:
            out.append(rf'sum({nb},{var})')
        i = after
    return ''.join(out)


def _nl_diff(src):
    """``\\frac{d}{dX}`` / ``\\frac{d^{n}}{dX^{n}}`` + ``(BODY)`` -> ``diff(BODY, X[, n])``."""
    out, i = [], 0
    pat = re.compile(r'\\frac\{d(?:\^\{?(\w+)\}?)?\}\{d\s*(\w)\}')
    while True:
        match = pat.search(src, i)
        if not match:
            out.append(src[i:])
            break
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        order, var = match.group(1), match.group(2)
        tail = rf',{order}' if order else ''
        out.append(rf'diff({_normalize_latex_ops(body)},{var}{tail})')
        i = after
    return ''.join(out)


def _nl_partial(src):
    """``\\partial_{X} (BODY)`` -> ``diff(BODY, X)``."""
    out, i = [], 0
    pat = re.compile(r'\\partial_\{(\w)\}')
    while True:
        match = pat.search(src, i)
        if not match:
            out.append(src[i:])
            break
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        out.append(rf'diff({_normalize_latex_ops(body)},{match.group(1)})')
        i = after
    return ''.join(out)


def _nl_integral(src):
    """``\\int [_{a}] [^{b}] (BODY) \\, dX`` -> ``int(BODY, X[, a, b])``."""
    out, i = [], 0
    pat = re.compile(r'\\int(?![A-Za-z])(?:\s*_\{([^{}]*)\})?(?:\s*\^\{([^{}]*)\})?')
    while True:
        match = pat.search(src, i)
        if not match:
            out.append(src[i:])
            break
        low = (match.group(1) or '').strip()
        up = (match.group(2) or '').strip()
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        mvar = re.match(r'(?:[\s\\,]*)d\s*(\w)', src[after:])
        var = mvar.group(1) if mvar else 'x'
        consume_end = after + (mvar.end() if mvar else 0)
        out.append(src[i:match.start()])
        nb = _normalize_latex_ops(body)
        if up != '':
            out.append(rf'int({nb},{var},{low or 0},{up})')
        else:
            out.append(rf'int({nb},{var})')
        i = consume_end
    return ''.join(out)


def _nl_limit(src):
    """``\\lim_{X \\to a} (BODY)`` -> ``limit(BODY, X, a)``."""
    out, i = [], 0
    pat = re.compile(r'\\lim\s*_\{([^{}]*?)\\to([^{}]*?)\}')
    while True:
        match = pat.search(src, i)
        if not match:
            out.append(src[i:])
            break
        body, after = _grab_balanced_paren(src, match.end())
        if body is None:
            out.append(src[i:match.end()])
            i = match.end()
            continue
        out.append(src[i:match.start()])
        var = match.group(1).strip() or 'x'
        point = match.group(2).strip() or '0'
        out.append(rf'limit({_normalize_latex_ops(body)},{var},{point})')
        i = after
    return ''.join(out)


def _build_op_data(name, args):
    """Assemble the per-op data payload from a function name + positional args
    (the same assembly the single-operator path uses, factored for reuse)."""
    op, fields = _AUTO_FUNCS[name]
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
        if op == 'diff' and int(str(data.get('order', '1') or '1')) > 1:
            align, value = _diff_steps_latex(
                data.get('expression', ''), Symbol(str(data.get('variable', 'x') or 'x')),
                int(str(data.get('order', '1') or '1')))
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
    if op in ('invert', 'ref'):
        try:
            ok, res = _dispatch_compute(op, data)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f'{op} 计算失败: {exc}', None
        if not ok:
            return False, res if isinstance(res, str) else str(res), None
        try:
            from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
            mat = CommonMatrixCalculator.parse_matrix_input(data.get('expression', ''))
            value = mat.inv() if op == 'invert' else mat.rref()[0]
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
        text = text[:loc['start']] + '(' + str(value) + ')' + text[loc['end']:]
        text = _normalize_latex_ops(text)
        if text in seen:
            return False, '计算未推进（可能存在循环定义）'
        seen.add(text)
        blocks.append((loc['raw'], steps, value, text))

    try:
        pre = cached_sympify(text)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, f'初等合并失败: {exc}'
    final = pre.doit() if hasattr(pre, 'doit') else pre
    return True, _render_compound(original_display, blocks, final)


def _render_compound(original_display, blocks, final_value):
    """One combined ``align``: original -> each operator's nested sub-steps ->
    the values substituted in -> final elementary merge. ``blocks`` are tuples
    ``(raw_call, steps_aligned, value, substituted_so_far_text)``."""
    lines = ['& ' + original_display]
    for call_raw, steps_latex, _value, _snap in blocks:
        if steps_latex:
            note = rf'\text{{算子 {call_raw} 的分步}}'
            lines.append(r'& \quad ' + note + r'\; ' + steps_latex)
    # the expression with every operator replaced by its computed value (verbatim)
    last_snap = blocks[-1][3] if blocks else ''
    if last_snap:
        lines.append(r'&= \text{' + last_snap + r'} \quad \text{代入各算子结果}')
    lines.append(r'&= \boxed{' + latex(final_value) + '}')
    return r'\begin{align}' + r' \\ '.join(lines) + r'\end{align}'


# Ops whose engine wants the RAW argument text (its parser handles nested symbolic
# names like Inverse/Add itself) -- these must keep the single-call fast path.
_RAW_ARG_OPS = {
    'matrix_symbolic', 'ref', 'operations', 'invert', 'rank', 'LU', 'diag', 'svd',
    'singular', 'eigen', 'schur', 'det', 'orthogonal', 'projection', 'dependence',
    'linear-system', 'linear-solver', 'transform-1', 'transform-2',
}
# Ops whose engine consumes a RESOLVED value, so a nested operator is worth hoisting.
_CALC_COMPOSABLE = {'diff', 'integral', 'limit', 'expr'}


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


def _dispatch_auto(expression: str):
    """Universal single-box dispatcher (Maple-like): classify by operator/structure.

    Recognizes function-style operators (det/rank/integral/diff/taylor/...) for every
    supported operation, plus structural forms: d/dx, integral, limit, sum, matrix
    literals (A+B / A*B / lone matrix), partial (∂), and uppercase-symbol products
    (symbolic matrices). Delegates to the existing per-op dispatch so all step-by-step
    engines are reused."""
    from sympy import Derivative, Integral, Limit, Sum  # pylint: disable=import-outside-toplevel

    # Any unfilled placeholder box -> one clear message, checked on the raw input
    # before conversion so latex_to_nested never trips over the empty slots.
    if re.search(r'\\square\b|\\Box\b|\\bigbox\b|\\placeholder\b', expression):
        return False, '请先填写输入框中的占位（□）后再计算'
    text = latex_to_nested(expression)  # bmatrix blocks -> [[...]] (no-op otherwise)

    # 0) compound: an operator call embedded inside a larger elementary expression
    #    (3*diff(x,x), sin(x)+int(x^2,x), diff(int(x^2,x),x)) is resolved recursively
    #    by reusing each operator's own engine + steps, splicing values back. A lone
    #    top-level operator is left to the fast path below (byte-for-byte unchanged).
    norm = _normalize_latex_ops(text)
    calls = _iter_op_calls(norm)
    if _should_compound(norm, calls):
        return _dispatch_compound(expression, norm)

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
                op = spec[0]
                return _dispatch_compute(op, _build_op_data(name, args))

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
            order = int(str(data.get('order', '1') or '1'))
            if order > 1:  # higher derivative: run the engine once per order, feeding back
                align, _val = _diff_steps_latex(expression, Symbol(variable), order)
                return True, align
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

        if operation_type == 'trace':
            return _matrix_trace(expression)

        if operation_type in ('nullspace', 'colspace', 'rowspace'):
            return _matrix_subspace(expression, operation_type)

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
                    simplified = sympify(expression).simplify()
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
