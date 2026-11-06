"""LaTeX ``^`` parsing must yield the same canonical structure as plain ``**``.

``parse_latex`` left nested sums unevaluated, e.g. ``1/((x**2+2*x)+5)`` instead of
``1/(x**2+2*x+5)``. Structural rule matching then missed simple rules (the integral
case is covered in ``test_integral_canonical.py``).
"""

from sympy import Add, Eq, exp, pi, symbols

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


def test_plain_caret_input_does_not_split_multi_letter_names():
    """Regression: '^' used to force parse_latex, which read pi*x as i*p."""
    x = symbols('x')
    assert cached_sympify('pi*x^2') == pi * x ** 2
    alpha = symbols('alpha')
    assert cached_sympify('alpha*x^2') == alpha * x ** 2
    beta = symbols('beta')
    assert cached_sympify('beta*x') == beta * x


def test_plain_e_power_is_exponential():
    x = symbols('x')
    assert cached_sympify('e^x') == exp(x)
    assert cached_sympify('2e^x') == 2 * exp(x)
    assert cached_sympify('xe^x') == x * exp(x)
    assert cached_sympify('e^(x+1)') == exp(x + 1)


def test_equation_input_still_parses_as_latex():
    x, y = symbols('x y')
    assert cached_sympify('x^2+y^2=1') == Eq(x ** 2 + y ** 2, 1)
