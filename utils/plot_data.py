"""Sample a single-variable scalar function y=f(x) for SVG plotting, WITHOUT eval.

Parse ONLY via ``cached_sympify`` (which carries the huge-power guard); build the
variable as a bare ``Symbol`` (with no assumptions — an over-constrained symbol
silently no-ops ``.subs``), evaluate on a fixed Float grid, coerce to ``float`` guarded
by ``math.isfinite`` + a magnitude cap, and emit ``None`` holes at singularities /
non-real points so the client can break the polyline at asymptotes. Optional tangent
overlay is computed from the same subs math. Pure (no Flask) so it is unit-testable.
"""

import math
import re

from sympy import Derivative, Expr, Float, Integral, Limit, N, Symbol

from config import (PLOT_COORD_DIGITS, PLOT_DEFAULT_SAMPLES, PLOT_DEFAULT_WINDOW,
                    PLOT_MAX_ABS_Y, PLOT_MAX_SAMPLES)
from utils.cache_utils import cached_sympify


def _to_float(expr, var, xval):
    """f(xval) as a finite float within magnitude cap, else None (a polyline break)."""
    try:
        # subs() itself can raise (e.g. factorial(-3) -> gamma pole, exp towers
        # overflowing); the substitution must be inside the guard too.
        val = expr.subs(var, xval)
        f = float(N(val))
    except Exception:  # pylint: disable=broad-exception-caught
        # SymPy's numeric conversion can raise OverflowError (huge exp towers),
        # RecursionError (floor/sign derivatives), or complex TypeError; a plot
        # must degrade to a gap, never turn into a 500.
        return None
    if not math.isfinite(f) or abs(f) > PLOT_MAX_ABS_Y:
        return None
    return round(f, PLOT_COORD_DIGITS)


def _as_num(value):
    r"""A plain int/float/numeric-string, else None (rejects symbols like \pi)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _eval_float(expr, var, xval):
    """f(xval) as a finite float, else None. Numeric failures never escape."""
    try:
        val = expr.subs(var, Float(repr(xval)))
        f = float(N(val))
    except Exception:  # pylint: disable=broad-exception-caught
        return None
    return f if math.isfinite(f) else None


def _find_numerical_roots(expr, var, lo, hi, *, max_roots=12, grid=400):
    """Approximate the real roots of f(x)=0 on (lo, hi).

    A grid sweep brackets sign changes; each bracket is bisected to sub-pixel
    width. A root of even multiplicity never changes sign, so it is caught as a
    local dip of |f| and located by a golden-section search. Both acceptance
    tests are RELATIVE to the curve's own magnitude: an absolute epsilon would
    report every grid point of a uniformly small function (f = 1e-5) as a root,
    and would miss a tangency whose dip does not land exactly on a grid point.
    """
    if not hi > lo:
        return []
    span = hi - lo
    step = span / (grid - 1)
    samples = []
    for i in range(grid):
        x = lo + i * step
        samples.append((x, _eval_float(expr, var, x)))

    finite = [abs(f) for _, f in samples if f is not None]
    if not finite:
        return []
    scale = max(finite)
    if scale == 0.0:
        # f vanishes across the whole window: report a single representative.
        return [{'x': round(lo + span / 2, PLOT_COORD_DIGITS), 'y': 0.0}]
    eps = scale * 1e-6

    def _bisect(a, b, fa):
        """|f| -> 0 on a bracket with fa*fb <= 0."""
        for _ in range(24):
            m = 0.5 * (a + b)
            fm = _eval_float(expr, var, m)
            if fm is None:
                return None
            if fm == 0.0 or abs(fm) <= 1e-12 * scale:
                return m
            if (fa < 0.0) == (fm < 0.0):
                a, fa = m, fm
            else:
                b = m
        return 0.5 * (a + b)

    def _golden_min(a, b):
        """Locate the local minimum of |f| inside [a, b] (derivative-free)."""
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - ratio * (b - a)
        d = a + ratio * (b - a)
        fc, fd = _eval_float(expr, var, c), _eval_float(expr, var, d)
        if fc is None or fd is None:
            return None
        fc, fd = abs(fc), abs(fd)
        for _ in range(24):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - ratio * (b - a)
                fc = _eval_float(expr, var, c)
                if fc is None:
                    return None
                fc = abs(fc)
            else:
                a, c, fc = c, d, fd
                d = a + ratio * (b - a)
                fd = _eval_float(expr, var, d)
                if fd is None:
                    return None
                fd = abs(fd)
        return 0.5 * (a + b)

    candidates = []
    for i, (x, f) in enumerate(samples):
        if f is None:
            continue
        if f == 0.0:
            candidates.append(x)
            continue
        if i + 1 < grid:
            xn, fn = samples[i + 1]
            if fn is not None and (f < 0.0) != (fn < 0.0):
                r = _bisect(x, xn, f)
                if r is not None:
                    candidates.append(r)
        if 1 <= i < grid - 1:
            xp, fp = samples[i - 1]
            xn, fn = samples[i + 1]
            if (fp is not None and fn is not None and abs(f) <= 0.01 * scale
                    and abs(f) < abs(fp) and abs(f) < abs(fn)):
                r = _golden_min(xp, xn)
                if r is not None:
                    candidates.append(r)

    dedupe = max(0.5 * step, 1e-12)
    picked = []
    for r in candidates:
        if any(abs(r - p) <= dedupe for p in picked):
            continue
        fr = _eval_float(expr, var, r)
        if fr is None or abs(fr) > max(eps, 1e-9 * scale):
            continue
        picked.append(r)
        if len(picked) >= max_roots:
            break

    res = []
    for r in picked:
        yr = _eval_float(expr, var, r)
        res.append({
            'x': round(r, PLOT_COORD_DIGITS),
            'y': round(yr, PLOT_COORD_DIGITS) if yr is not None else 0.0,
        })
    return res


def sample_curve(data):
    """Sample ``data['expression']`` over x and return JSON-safe points.

    Accepts ``expression, variable`` and optional numeric ``xmin/xmax`` (or
    ``lower/upper``), ``samples``, and ``tangent_at`` for a tangent-line overlay.
    Returns ``{success, variable, xmin, xmax, points:[[x, y|null], ...], ...}``."""
    expr_str = str(data.get('expression', '')).strip()
    if not expr_str:
        return {'success': False, 'error': '缺少表达式'}
    try:
        expr = cached_sympify(expr_str)
    except Exception as e:  # pylint: disable=broad-exception-caught
        return {'success': False, 'error': f'无法解析表达式: {e}'}
    if not isinstance(expr, Expr):
        return {'success': False, 'error': '仅支持单变量标量函数作图'}
    if expr.has(Limit, Derivative, Integral):
        # Numeric evaluation of these is pathologically slow on a synchronous path.
        return {'success': False, 'error': '该表达式含未求值的极限/导数/积分, 暂不支持作图'}

    var = None
    name = str(data.get('variable', '') or '').strip()
    if name and Symbol(name) in expr.free_symbols:
        var = Symbol(name)
    if var is None:
        free = list(expr.free_symbols)
        var = free[0] if len(free) == 1 else None
    if var is None:
        return {'success': False, 'error': '需要恰好一个自变量'}
    if expr.free_symbols - {var}:
        hint = ''
        if re.match(r'\s*\\?[A-Za-z][A-Za-z0-9]*\s*\(', expr_str):
            hint = '；输入看起来是算子形式（如 diff(...)），作图请只填写函数本身，例如 x^2-1'
        return {'success': False, 'error': '存在多个自由符号, 无法作为单变量函数作图' + hint}

    lo = _as_num(data.get('xmin', data.get('lower')))
    hi = _as_num(data.get('xmax', data.get('upper')))
    span = 2 * PLOT_DEFAULT_WINDOW
    if lo is None and hi is None:
        lo, hi = -PLOT_DEFAULT_WINDOW, PLOT_DEFAULT_WINDOW
    elif lo is None:
        # Only an upper bound was given: keep the default window width.
        lo = hi - span
    elif hi is None:
        # Only a lower bound was given: previously this produced a *reversed*
        # window ([100 .. 10]) because hi silently fell back to +10.
        hi = lo + span
    if hi <= lo:
        return {'success': False, 'error': 'xmin 必须小于 xmax'}
    try:
        n = int(data.get('samples', PLOT_DEFAULT_SAMPLES))
    except (TypeError, ValueError):
        n = PLOT_DEFAULT_SAMPLES
    n = max(2, min(n, PLOT_MAX_SAMPLES))

    step = (hi - lo) / (n - 1)
    xs = [lo + i * step for i in range(n)]
    points = [[round(xv, PLOT_COORD_DIGITS), _to_float(expr, var, Float(repr(xv)))]
              for xv in xs]

    out = {'success': True, 'variable': str(var),
           'xmin': round(lo, PLOT_COORD_DIGITS), 'xmax': round(hi, PLOT_COORD_DIGITS),
           'points': points}
    finite = [p[1] for p in points if p[1] is not None]
    if finite:
        out['ymin'] = round(min(finite), PLOT_COORD_DIGITS)
        out['ymax'] = round(max(finite), PLOT_COORD_DIGITS)
    else:
        out['ymin'], out['ymax'] = -1.0, 1.0

    a = _as_num(data.get('tangent_at'))
    if a is not None and lo <= a <= hi:
        try:
            slope = float(N(expr.diff(var).subs(var, Float(repr(a)))))
            y0 = float(N(expr.subs(var, Float(repr(a)))))
        except Exception:  # pylint: disable=broad-exception-caught
            # floor/sign derivatives recurse inside evalf; exp towers overflow.
            slope = y0 = None
        if slope is not None and math.isfinite(slope) and math.isfinite(y0):
            out['tangent'] = {
                'x': round(a, PLOT_COORD_DIGITS), 'y': round(y0, PLOT_COORD_DIGITS),
                'slope': round(slope, PLOT_COORD_DIGITS),
                'y1': round(y0 + slope * (lo - a), PLOT_COORD_DIGITS),
                'y2': round(y0 + slope * (hi - a), PLOT_COORD_DIGITS),
            }

    try:
        roots = _find_numerical_roots(expr, var, lo, hi)
        if roots:
            out['roots'] = roots
    except Exception:
        pass

    return out
