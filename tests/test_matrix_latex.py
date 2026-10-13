"""Tests for MathLive LaTeX matrix -> nested-list conversion and matrix ops."""

from routes.compute_service import start_compute
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
