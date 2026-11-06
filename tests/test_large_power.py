"""Guard: huge integer-power literals stay unevaluated (render instead of crashing)."""

from sympy import Symbol, latex

from routes.compute_service import start_compute
from utils.cache_utils import _has_huge_power_literal, cached_sympify

x = Symbol('x')


def test_detection_threshold():
    assert _has_huge_power_literal('2**1000000') is True
    assert _has_huge_power_literal('x**3') is False
    assert _has_huge_power_literal('2**500') is False


def test_nested_towers_are_detected():
    assert _has_huge_power_literal('2**10**9') is True
    assert _has_huge_power_literal('2**9**9') is True
    assert _has_huge_power_literal('9**9') is False
    assert _has_huge_power_literal('3**5**3') is False
    assert _has_huge_power_literal('2**5**5') is False
    assert _has_huge_power_literal('2**(10**9)') is True
    assert _has_huge_power_literal('2**(3**5**5)') is True
    assert _has_huge_power_literal('2**(9**9)') is True
    assert _has_huge_power_literal('2**(1000000*1000000)') is True
    assert _has_huge_power_literal('2**(1000000/1000)') is False
    assert _has_huge_power_literal('2**(x*10**9)') is False
    assert _has_huge_power_literal('2**(10+1)') is False


def test_huge_power_stays_symbolic_and_renders():
    rendered = latex(cached_sympify('x + 2**1000000'))
    assert '2^' in rendered
    assert 'x' in rendered


def test_nested_towers_stay_unevaluated_and_render():
    # Regression: these used to be fed to evaluate=True parsing and computed
    # (or hung). They must parse instantly as unevaluated powers.
    for text in ('2**10**9', '2**(10**9)', '2**9**9', '2**(1000000*1000000)'):
        rendered = latex(cached_sympify(text))
        assert '^{' in rendered or '^' in rendered, text


def test_normal_parse_unaffected():
    assert cached_sympify('x**3 + 2*x') == x**3 + 2 * x


def test_huge_power_failure_gets_a_readable_message():
    """Evaluating/printing a huge power trips CPython's 4300-digit limit.

    The raw 'Exceeds the limit (4300 digits)...' text used to be shown to the
    user; the compute path must translate it into an actionable Chinese message.
    """
    ok, res = start_compute('integral', {
        'operation_type': 'integral', 'expression': '2**1000000',
        'variable': 'x'})
    assert not ok
    assert '幂次或整数过大' in str(res), res
    assert '4300' not in str(res), res
