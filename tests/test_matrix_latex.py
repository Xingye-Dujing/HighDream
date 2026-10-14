"""Tests for MathLive LaTeX matrix -> nested-list conversion and matrix ops."""

from routes.compute_service import start_compute, _dispatch_auto
from utils.matrix_latex import latex_to_nested


def test_converter_bmatrix():
    s = r'\begin{bmatrix}1&2\\3&4\end{bmatrix}'
    assert latex_to_nested(s) == '[[1,2],[3,4]]'


def test_converter_pmatrix_and_latex_cells():
    s = r'\begin{pmatrix}\frac{1}{2}&0\\0&\sqrt{2}\end{pmatrix}'
    assert latex_to_nested(s) == '[[1/2,0],[0,sqrt(2)]]'


def test_converter_preserves_newline_for_two_matrices():
    s = r'\begin{bmatrix}1&0\\0&1\end{bmatrix}' + '\n' + r'\begin{bmatrix}2&0\\0&2\end{bmatrix}'
    out = latex_to_nested(s)
    assert out == '[[1,0],[0,1]]\n[[2,0],[0,2]]'


def test_converter_noop_for_plain_text():
    assert latex_to_nested('[[1,2],[3,4]]') == '[[1,2],[3,4]]'


def test_converter_handles_mathlive_emission_quirk():
    # MathLive sometimes drops the backslash on \end; converter must still parse.
    s = r'\begin{bmatrix}1&2\\3&4end{bmatrix}'
    assert latex_to_nested(s) == '[[1,2],[3,4]]'


def test_det_accepts_bmatrix_latex():
    s = r'\begin{bmatrix}1&2\\3&4\end{bmatrix}'
    ok, res = start_compute('det', {'operation_type': 'det', 'expression': s, 'type': '1'})
    assert ok and '-2' in str(res)


def test_det_symbolic_pivot_no_crash():
    # det with a symbolic pivot: the row-op *description* used to compare `factor_ < 0`,
    # which raises "cannot determine truth value" on a symbolic factor (e.g. 1/a).
    # det must still compute the symbolic value instead of erroring.
    ok, res = _dispatch_auto(r'det([[a,1],[1,a]])')
    assert ok, res
    s = res if isinstance(res, str) else str(res)
    assert 'a^{2} - 1' in s
    ok2, _ = _dispatch_auto(r'det([[1,a],[a,1]])')
    assert ok2
