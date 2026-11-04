# -*- coding: utf-8 -*-
"""Tests for the SVG-plot sampler (utils/plot_data.sample_curve) — pure logic.

Covers the safety contract: eval-free parsing, null holes at asymptotes / out-of-domain,
hard sample clamp, multi-symbol and non-scalar rejection, magnitude caps, and the
tangent overlay computed from the same substitution math."""

from config import PLOT_MAX_SAMPLES
from utils.plot_data import sample_curve


def _ys(res):
    return [p[1] for p in res['points']]


def test_hole_at_asymptote():
    res = sample_curve({'expression': '1/x', 'variable': 'x', 'xmin': -2, 'xmax': 2, 'samples': 5})
    assert res['success'] is True
    ys = _ys(res)
    assert ys[2] is None                 # x = 0 -> singular, emitted as a break
    assert all(v is not None for v in (ys[0], ys[1], ys[3], ys[4]))


def test_out_of_domain_is_null():
    res = sample_curve({'expression': 'log(x)', 'variable': 'x', 'xmin': -2, 'xmax': 4, 'samples': 7})
    ys = _ys(res)
    assert all(v is None for v in ys[:3])   # x<0: log undefined -> holes
    assert any(v is not None for v in ys[3:])


def test_samples_are_clamped():
    res = sample_curve({'expression': 'x**2', 'variable': 'x', 'samples': 10 ** 9})
    assert len(res['points']) == PLOT_MAX_SAMPLES


def test_rejects_multiple_free_symbols():
    res = sample_curve({'expression': 'x + y', 'variable': 'x'})
    assert res['success'] is False


def test_huge_power_does_not_raise():
    # cached_sympify's huge-power guard + float overflow -> holes, never a crash/hang
    res = sample_curve({'expression': 'x**1000000', 'variable': 'x', 'xmin': 0, 'xmax': 2, 'samples': 20})
    assert res['success'] is True


def test_one_sided_window_keeps_direction():
    # Only a lower bound: the window must run forwards, not [100 .. 10].
    res = sample_curve({'expression': 'x', 'xmin': 100, 'samples': 5})
    assert res['success'] is True
    xs = [p[0] for p in res['points']]
    assert xs == sorted(xs), xs
    assert res['xmax'] > 100

    # Only an upper bound mirrors it backwards.
    res = sample_curve({'expression': 'x', 'xmax': 5, 'samples': 5})
    xs = [p[0] for p in res['points']]
    assert xs == sorted(xs), xs
    assert res['xmin'] < 5


def test_reversed_window_is_rejected():
    res = sample_curve({'expression': 'x', 'xmin': 5, 'xmax': 1})
    assert res['success'] is False
    assert 'xmin' in res['error']


def test_tangent_overlay():
    res = sample_curve({'expression': 'x**2', 'variable': 'x', 'xmin': -1, 'xmax': 1,
                        'samples': 3, 'tangent_at': 0.5})
    t = res['tangent']
    assert abs(t['slope'] - 1.0) < 1e-9
    assert abs(t['y'] - 0.25) < 1e-9       # f(0.5)=0.25
    assert abs(t['y1'] - (0.25 + 1.0 * (-1 - 0.5))) < 1e-9


def _root_xs(res):
    return [r['x'] for r in res.get('roots', [])]


def test_roots_are_bracketed_and_sorted():
    res = sample_curve({'expression': 'x**2 - 1', 'variable': 'x', 'xmin': -2, 'xmax': 2})
    xs = _root_xs(res)
    assert len(xs) == 2
    assert abs(xs[0] + 1.0) < 1e-3
    assert abs(xs[1] - 1.0) < 1e-3


def test_even_multiplicity_root_is_found_non_grid_aligned():
    # Tangency between two grid points: no sign change, so it needs the |f| dip path.
    res = sample_curve({'expression': '(x - 0.5)**2', 'variable': 'x', 'xmin': -2, 'xmax': 2})
    xs = _root_xs(res)
    assert len(xs) == 1
    assert abs(xs[0] - 0.5) < 1e-2


def test_no_roots_when_none_exist():
    res = sample_curve({'expression': 'x**2 + 1', 'variable': 'x', 'xmin': -3, 'xmax': 3})
    assert res.get('roots', []) == []


def test_uniformly_small_function_is_not_all_roots():
    # |f| < 3e-4 everywhere must not turn every grid point into a root.
    res = sample_curve({'expression': '1/100000', 'variable': 'x', 'xmin': -10, 'xmax': 10})
    assert res.get('roots', []) == []


def test_asymptote_is_not_a_root():
    res = sample_curve({'expression': '1/(x - 0.3)', 'variable': 'x', 'xmin': -2, 'xmax': 2})
    assert res.get('roots', []) == []
