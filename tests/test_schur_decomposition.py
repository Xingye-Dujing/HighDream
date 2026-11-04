"""Schur decomposition regressions.

The old iterative QR method crashed on matrices with repeated eigenvalues
(``list(eigenvalues.keys())[i]`` indexed distinct eigenvalues by row number) and
on singular matrices (SymPy's QR drops zero columns, so ``Q_sub`` was not square).
It also left T non-triangular. Every case here must return an orthogonal Q and a
block-upper-triangular T with ``Q T Q^H == A``.
"""

import pytest
from sympy import Matrix, eye, simplify, sympify, zeros

from domains import SchurDecomposition

# repeated eigenvalues, zero matrix, rank-deficient, distinct real, complex pair.
CASES = [
    '[[1,0],[0,1]]',
    '[[2,0],[0,2]]',
    '[[0,0],[0,0]]',
    '[[1,2,3],[4,5,6],[7,8,9]]',
    '[[1,2],[3,4]]',
    '[[4,1],[2,3]]',
    '[[1,2],[2,1]]',
    '[[0,-1],[1,0]]',
]


@pytest.mark.parametrize('expr', CASES)
def test_schur_has_orthogonal_q_and_reconstructs(expr):
    Q, T = SchurDecomposition().auto_schur_decomposition(expr)
    A = Matrix(sympify(expr))
    assert simplify(Q.T.conjugate() * Q - eye(A.rows)) == zeros(A.rows)
    assert simplify(Q * T * Q.T.conjugate() - A) == zeros(A.rows)


@pytest.mark.parametrize('expr', CASES)
def test_schur_t_is_block_upper_triangular(expr):
    # Real eigenvalues give a strictly upper-triangular T; a complex-conjugate
    # pair keeps a 2x2 block, so only the entries below the first subdiagonal
    # must vanish.
    _, T = SchurDecomposition().auto_schur_decomposition(expr)
    n = T.rows
    for i in range(n):
        for j in range(i - 1):
            assert simplify(T[i, j]) == 0, (expr, i, j, T)


def test_schur_repeated_eigenvalue_is_diagonal():
    _, T = SchurDecomposition().auto_schur_decomposition('[[2,0],[0,2]]')
    assert simplify(T - 2 * eye(2)) == zeros(2)


def test_schur_identity_is_identity():
    _, T = SchurDecomposition().auto_schur_decomposition('[[1,0],[0,1]]')
    assert simplify(T - eye(2)) == zeros(2)
