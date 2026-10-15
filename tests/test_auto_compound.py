# -*- coding: utf-8 -*-
"""Pure-logic regression tests for the universal-box recursive compound evaluator
(routes.compute_service._dispatch_auto -> _dispatch_compound)."""

from routes.compute_service import (
    _dispatch_auto, _dispatch_compound, _find_innermost_op, _iter_op_calls,
    _normalize_latex_ops,
)


def test_normalize_latex_diff_consumes_body():
    # the body paren must be consumed (no duplicated "(x^2)")
    assert _normalize_latex_ops(r'3*\frac{d}{d x}(x^2)') == '3*diff(x^2,x)'


def test_normalize_latex_integral_consumes_differential():
    assert _normalize_latex_ops(r'\int (x)\, d x') == 'int(x,x)'


def test_normalize_latex_limit():
    assert _normalize_latex_ops(r'\lim_{x \to 0}(x)') == 'limit(x,x,0)'


def test_find_innermost_prefers_nested():
    loc = _find_innermost_op('diff(int(x^2,x),x)')
    assert loc is not None and loc['name'] == 'int'


def test_iter_op_calls_ignores_plain_functions():
    # sin/exp/foo are NOT operators -> not scanned as calls
    assert not _iter_op_calls('sin(x)+2')


def test_compound_scalar_derivative():
    ok, res = _dispatch_auto('3*diff(x,x)')
    assert ok
    assert '\\begin{align}' in res and '\\boxed{3}' in res
    assert 'diff(x,x)' in res  # the operator's own sub-derivation is shown


def test_compound_nested_integral_then_derivative():
    ok, res = _dispatch_compound('diff(int(x^2,x),x)', 'diff(int(x^2,x),x)')
    assert ok
    assert '\\boxed{x^{2}}' in res


def test_compound_mixed_latex_and_functional():
    ok, res = _dispatch_auto(r'\frac{d}{d x}(x^3)+\frac{d}{d x}(x^2)')
    assert ok
    assert '\\boxed{3 x^{2} + 2 x}' in res  # 3x^2 + 2x


def test_single_operator_path_unchanged():
    # a lone operator must NOT gain the compound wrapper (backward compatible)
    ok, res = _dispatch_auto('diff(x,x)')
    assert ok
    assert '\\boxed' not in res and '算子' not in res
    assert res.startswith('\\begin{align}&\\frac{d}')


def test_placeholder_guard_precedes_compound():
    ok, res = _dispatch_auto('3*\\frac{d}{d\\placeholder{}}(\\placeholder{})')
    assert not ok
    assert '占位' in res


def test_unknown_function_no_operator_error():
    ok, _ = _dispatch_auto('2*foo(x)')  # foo stays a plain function; not an error
    assert ok


def test_symbolic_matrix_stays_on_fast_path():
    # matrix_symbolic parses Inverse/Add itself -> must NOT be hijacked by compound
    ok, res = _dispatch_auto('msimplify(A*B*Inverse(B))')
    assert ok, res
    assert '算子' not in res and '\\boxed' not in res


def test_nested_calculus_hoists_inner_operator():
    # diff wrapping int -> compound resolves the inner integral first, then the
    # derivative; the result is FTOC x^2 and both operators show their own steps.
    ok, res = _dispatch_auto('diff(int(x^2,x),x)')
    assert ok, res
    assert '\\boxed{x^{2}}' in res
    assert 'int(x^2,x) 的分步' in res and 'diff(' in res


def test_scalar_linear_algebra_composes():
    # det -> scalar and rank -> int substitute into arithmetic (det=-2, rank=2 -> 0)
    ok, res = _dispatch_auto('det([[1,2],[3,4]]) + rank([[1,0],[0,1]])')
    assert ok, res
    assert '\\boxed{0}' in res


def test_matrix_valued_result_rejected_in_scalar_context():
    ok, res = _dispatch_auto('inverse([[1,2],[3,4]]) + 1')
    assert not ok
    assert '矩阵' in res


def test_det_scaled_by_constant():
    ok, res = _dispatch_auto('2*det([[2,0],[0,3]])')
    assert ok, res
    assert '\\boxed{12}' in res


def test_taylor_polynomial_composes():
    ok, res = _dispatch_auto('taylor(e^x,x,0,2) + 1')
    assert ok, res
    assert '\\boxed{\\frac{x^{2}}{2} + x + 2}' in res


def test_volume_application_composes():
    # volume (disk) -> pi/3, times 3 -> pi ; shown with its own integral steps
    ok, res = _dispatch_auto('volume(x,x,0,1)*3')
    assert ok, res
    assert '\\boxed{\\pi}' in res


def test_auto_limit_carries_one_sided_direction():
    assert _normalize_latex_ops(r'\lim_{x \to 0^{+}} (x)') == 'limit(x,x,0,+)'
    assert _normalize_latex_ops(r'\lim_{x \to 0^{-}} (x)') == 'limit(x,x,0,-)'
    assert _normalize_latex_ops(r'\lim_{x \to 0^{\pm}} (x)') == 'limit(x,x,0,+-)'
    # no side marker -> direction left unspecified (engine default)
    assert _normalize_latex_ops(r'\lim_{x \to 0} (x)') == 'limit(x,x,0)'


def test_auto_limit_functional_direction_ok():
    ok, _ = _dispatch_auto('limit((1-1/x)^x, x, 0, +)')
    assert ok
