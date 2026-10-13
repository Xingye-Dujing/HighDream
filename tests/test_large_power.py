"""Guard: huge integer-power literals stay unevaluated (render instead of crashing)."""

from sympy import Symbol, latex

from utils.cache_utils import _has_huge_power_literal, cached_sympify

x = Symbol('x')


def test_detection_threshold():
    assert _has_huge_power_literal('2**1000000') is True
    assert _has_huge_power_literal('x**3') is False
    assert _has_huge_power_literal('2**500') is False


def test_huge_power_stays_symbolic_and_renders():
    rendered = latex(cached_sympify('x + 2**1000000'))
    assert '2^' in rendered
    assert 'x' in rendered


def test_normal_parse_unaffected():
    assert cached_sympify('x**3 + 2*x') == x**3 + 2 * x
