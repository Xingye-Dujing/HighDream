"""Regression tests for symbolic matrix expression simplification."""

from sympy import Symbol
from sympy.matrices.expressions import MatrixSymbol

from domains.matrix.matrix_symbolic import simplify_matrix_symbolic

n = Symbol('n')
A = MatrixSymbol('A', n, n)


def _final(text):
    return simplify_matrix_symbolic(text)[-1][0]


def test_inverse_cancellation():
    assert _final('A*B*Inverse(B)') == A
    assert str(_final('A*Inverse(A)')) == 'I'


def test_double_inverse():
    assert _final('Inverse(Inverse(A))') == A


def test_inverse_of_product_reverses():
    assert str(_final('Inverse(A*B)')) == 'B**(-1)*A**(-1)'


def test_transpose_reversal():
    assert str(_final('Transpose(A*B)')) == 'B.T*A.T'


def test_left_distribution():
    assert str(_final('A*(B+C)')) == 'A*B + A*C'
