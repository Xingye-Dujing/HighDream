"""Regression tests for the hardened user-input parsing path.

``sympify``/``parse_expr`` evaluate generated Python code, so every HTTP-facing
entry point must stay on the builtin-free parsing path.
"""

import pytest
from sympy import Abs, sin, Symbol

from app import app
from utils.cache_utils import cached_sympify


@pytest.mark.parametrize('payload', [
    '__import__("os").getcwd()',
    'eval("1+1")',
    'exec("x=1")',
    'globals()',
    'open("/etc/passwd")',
    '().__class__',
    '(lambda: 1)()',
    '[x for x in range(3)]',
    'sin(x).__class__',
    'x.__class__.__mro__',
])
def test_code_execution_payloads_are_rejected(payload):
    """No user string may reach ``eval`` with live builtins."""
    with pytest.raises(ValueError, match='不允许的语法'):
        cached_sympify(payload)


def test_compute_payload_is_rejected_by_normalization():
    """The /api/compute input normalizer uses the same guard."""
    from core.input_validation import normalize_scalar_input  # pylint: disable=import-outside-toplevel
    ok, result = normalize_scalar_input('__import__("os").getcwd()')
    assert ok is False
    assert '不允许的语法' in result


def test_parse_endpoint_does_not_execute_python():
    client = app.test_client()
    resp = client.post('/api/parse', json={
        'expression': '__import__("os").getcwd()',
        'operation_type': 'arithmetic',
    })
    body = resp.get_json()
    assert body['success'] is False
    assert '不允许的语法' in body['error']


def test_matrix_parsers_reject_code_execution():
    """Every matrix entry point must use the guarded parser, not raw sympify."""
    from core.common_matrix_calculator import CommonMatrixCalculator  # pylint: disable=import-outside-toplevel
    from domains.matrix.matrix_analyzer import MatrixAnalyzer  # pylint: disable=import-outside-toplevel
    from domains.matrix.ref_calculator import RefCalculator  # pylint: disable=import-outside-toplevel

    payload = '[[__import__("os").getcwd()]]'
    with pytest.raises(ValueError, match='不允许的语法'):
        CommonMatrixCalculator.parse_matrix_input(payload)
    with pytest.raises(ValueError):
        RefCalculator._parse_matrix_input(payload)  # pylint: disable=protected-access
    with pytest.raises(ValueError):
        MatrixAnalyzer().analyze(payload)


def test_matrix_endpoints_reject_code_execution():
    client = app.test_client()
    payload = '[[__import__("os").getcwd()]]'
    for path, body in [
        ('/api/matrix_analyze', {'expression': payload}),
        ('/api/matrix_cal', {'expression': payload, 'operations': 'transpose'}),
        ('/api/get_process', {'type': 'det_1', 'expression': payload}),
    ]:
        response = client.post(path, json=body)
        assert response.status_code == 200, path
        result = response.get_json()
        assert result['success'] is False, path
        # getcwd() would leave the project path in the parse error; its absence
        # proves the payload was rejected before evaluation.
        assert 'HighDream' not in str(result), path


def test_operator_form_converter_does_not_execute_python():
    """The form toggle renders operands via the guarded parser (or leaves them as text)."""
    from routes.op_form import _L  # pylint: disable=import-outside-toplevel

    payload = '__import__("os").getcwd()'
    assert _L(payload) == payload


def test_math_semantics_are_preserved():
    """Hardening must not change the documented parse conventions."""
    x = Symbol('x')
    assert cached_sympify('xy') == Symbol('xy')  # never implicitly split
    assert cached_sympify('x y') == cached_sympify('x*y')
    assert cached_sympify('2x') == cached_sympify('2*x')
    assert cached_sympify('sin(x)') == sin(x)
    assert cached_sympify(r'\frac{1}{x}') == 1 / x
    assert str(cached_sympify('x + 2**1000000')).startswith('x + 2**1000000')
    # Pure builtins that sympify() normally exposes must keep working: without
    # ``abs`` the input abs(x)/x degraded to an undefined function.
    assert cached_sympify('abs(x)/x') == Abs(x) / x
    assert cached_sympify('min(x, 2)').has(Symbol('x'))
    assert cached_sympify('sum([1, 2, 3])') == 6


def test_normal_expressions_still_parse():
    assert cached_sympify('x**2 + 3*x + 1') == Symbol('x')**2 + 3 * Symbol('x') + 1
    assert cached_sympify('oo') == cached_sympify('infinity')


def test_trusted_flag_is_an_explicit_escape_hatch():
    """Internal SymPy source strings can opt out, but must say so explicitly."""
    assert cached_sympify('x**2', trusted=True) == Symbol('x')**2
