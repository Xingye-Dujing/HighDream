"""Sample a single-variable scalar function y=f(x) for SVG plotting, WITHOUT eval.

Parse ONLY via ``cached_sympify`` (which carries the huge-power guard); build the
variable as a bare ``Symbol`` (with no assumptions — an over-constrained symbol
silently no-ops ``.subs``), evaluate on a fixed Float grid, coerce to ``float`` guarded
by ``math.isfinite`` + a magnitude cap, and emit ``None`` holes at singularities /
non-real points so the client can break the polyline at asymptotes. Optional tangent
overlay is computed from the same subs math. Pure (no Flask) so it is unit-testable.
"""

import math

from sympy import Derivative, Expr, Float, Integral, Limit, N, Symbol

from config import (PLOT_COORD_DIGITS, PLOT_DEFAULT_SAMPLES, PLOT_DEFAULT_WINDOW,
                    PLOT_MAX_ABS_Y, PLOT_MAX_SAMPLES)
from utils.cache_utils import cached_sympify


def _to_float(expr, var, xval):
    """f(xval) as a finite float within magnitude cap, else None (a polyline break)."""
    val = expr.subs(var, xval)
    try:
        f = float(N(val))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or abs(f) > PLOT_MAX_ABS_Y:
        return None
    return round(f, PLOT_COORD_DIGITS)


def _as_num(value):
    """A plain int/float/numeric-string, else None (rejects symbols like \pi)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _eval_float(expr, var, xval):
    try:
        val = expr.subs(var, Float(repr(xval)))
        f = float(N(val))
        if not math.isfinite(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _find_numerical_roots(expr, var, lo, hi, max_roots=12, grid=400):
    """Find approximate real roots of f(x)=0 in (lo, hi) by sign changes + linear approx."""
    roots = []
    if hi - lo < 1e-12:
        return roots
    fp = None
    try:
        fp = expr.diff(var)
    except Exception:
        fp = None

    step = (hi - lo) / (grid - 1)
    prev_x = lo
    prev_f = _eval_float(expr, var, prev_x)
    for i in range(1, grid):
        x = lo + i * step
        f = _eval_float(expr, var, x)
        if prev_f is not None and f is not None:
            if prev_f == 0.0:
                roots.append(prev_x)
            elif f == 0.0:
                pass
            elif prev_f * f < 0.0:
                t = prev_f / (prev_f - f) if (prev_f - f) != 0 else 0.5
                xr = prev_x + t * (x - prev_x)
                for _ in range(8):
                    f2 = _eval_float(expr, var, xr)
                    if f2 is None:
                        break
                    if abs(f2) < 1e-9:
                        break
                    if fp is not None:
                        fp2 = _eval_float(fp, var, xr)
                        if fp2 is not None and abs(fp2) > 1e-12:
                            xr_new = xr - f2 / fp2
                            if not (lo - 1e-6 <= xr_new <= hi + 1e-6):
                                break
                            xr = xr_new
                            continue
                    break
                if lo <= xr <= hi:
                    fr2 = _eval_float(expr, var, xr)
                    if fr2 is not None and abs(fr2) < 2e-3:
                        roots.append(xr)
            elif abs(f) < 3e-4:
                roots.append(x)
        prev_x, prev_f = x, f

    out = []
    for r in roots:
        if not out or min(abs(r - o) for o in out) > 2e-3:
            out.append(r)
        if len(out) >= max_roots:
            break

    res = []
    for r in out:
        yr = _eval_float(expr, var, r)
        if yr is None:
            yr = 0.0
        res.append({
            'x': round(r, PLOT_COORD_DIGITS),
            'y': round(yr, PLOT_COORD_DIGITS),
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
        return {'success': False, 'error': '存在多个自由符号, 无法作为单变量函数作图'}

    lo = _as_num(data.get('xmin', data.get('lower')))
    hi = _as_num(data.get('xmax', data.get('upper')))
    if lo is None or hi is None or hi <= lo:
        lo = lo if lo is not None else -PLOT_DEFAULT_WINDOW
        hi = hi if hi is not None and hi > lo else PLOT_DEFAULT_WINDOW
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
        except (TypeError, ValueError):
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
