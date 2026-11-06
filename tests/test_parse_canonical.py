"""LaTeX ``^`` parsing must yield the same canonical structure as plain ``**``.

``parse_latex`` left nested sums unevaluated, e.g. ``1/((x**2+2*x)+5)`` instead of
``1/(x**2+2*x+5)``. Structural rule matching then missed simple rules (the integral
case is covered in ``test_integral_canonical.py``).
"""

from sympy import Add

from utils.cache_utils import cached_sympify


def test_latex_caret_flattens_nested_sums():
    caret = cached_sympify('1/(x^2+2*x+5)')
    plain = cached_sympify('1/(x**2+2*x+5)')
    assert caret == plain
    denominator = caret.args[0]
    assert isinstance(denominator, Add)
    assert len(denominator.args) == 3, denominator


def test_huge_braced_power_stays_unevaluated():
    expr = cached_sympify('2^{1000000}')
    assert expr.is_Pow
    assert expr.exp == 1000000
    assert expr.base == 2
