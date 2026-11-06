"""Regression tests for issues found in the project review.

Each test pins down the behaviour that used to be wrong or to crash:
- singular "special" matrices used to get a pseudo inverse full of ``zoo``/``oo``
- the silent (no step output) rank elimination used an unnormalized pivot and
  counted eigenvalues without multiplicity
- the Weierstrass-substitution gate accepted non-integer powers
- a parametric definite integral (e.g. ``∫a dx``) was mislabelled as divergent
- user-supplied workload sizes (order/depth/L'Hôpital count) were unclamped
"""

from sympy import Symbol, sqrt, sin, Matrix

from domains.matrix.rank import Rank
from routes.compute_service import start_compute
from utils.elementary_expression import can_use_weierstrass

x = Symbol('x')


def test_singular_diagonal_special_matrix_inverse_rejected():
    ok, res = start_compute('invert', {'operation_type': 'invert', 'expression': '[[2,0],[0,0]]'})
    assert ok is False
    assert '不可逆' in str(res)


def test_nonsingular_diagonal_special_matrix_inverse_ok():
    ok, _ = start_compute('invert', {'operation_type': 'invert', 'expression': '[[2,0],[0,3]]'})
    assert ok is True


def test_rank_silent_row_echelon_accepts_unnormalized_pivot():
    # show_steps=False used to skip pivot normalization, so [[2,4],[1,2]] (rank 1)
    # was reported as rank 2.
    assert Rank().rank_by_row_echelon(Matrix([[2, 4], [1, 2]]), show_steps=False) == 1
    # The API rank of [[2,0,0],[0,2,1],[0,0,0]] is 2 and every method agrees.
    ok, res = start_compute('rank', {'operation_type': 'rank', 'expression': '[[2,0,0],[0,2,1],[0,0,0]]'})
    assert ok
    assert '= 2' in str(res)


def test_rank_eigenvalues_counts_multiplicity():
    assert Rank().rank_by_eigenvalues(Matrix([[2, 0, 0], [0, 2, 1], [0, 0, 0]])) == 2
    # A nilpotent Jordan block is not diagonalizable: this method must decline,
    # never report 0.
    assert Rank().rank_by_eigenvalues(Matrix([[0, 1], [0, 0]])) is None


def test_geometric_integrals_rejects_non_integer_weierstrass_power():
    assert can_use_weierstrass(sqrt(sin(x)), x) is False
    assert can_use_weierstrass(sin(x) ** 2, x) is True
    assert can_use_weierstrass(sin(x) ** -1, x) is True


def test_parametric_definite_integral_not_mislabelled_divergent():
    ok, res = start_compute('integral', {
        'operation_type': 'integral', 'expression': 'a',
        'variable': 'x', 'lower': '0', 'upper': '1'})
    assert ok
    assert '发散' not in str(res)


def test_diff_order_is_clamped():
    # order=10**9 used to loop the diff engine billions of times; it is clamped
    # server-side to HARD_MAX_DIFF_ORDER and must return promptly.
    ok, _ = start_compute('diff', {'operation_type': 'diff', 'expression': 'x**3', 'variable': 'x', 'order': '1000000000'})
    assert ok


def test_repeated_eigenvalue_multiplicity_and_diagonalizability():
    """Identity has λ=1 with algebraic multiplicity 2, not 1.

    ``solve`` returned each distinct root once, so the summary said algebraic 1
    vs geometric 2 and concluded "not diagonalizable" for the identity matrix.
    """
    ok, res = start_compute('eigen', {
        'operation_type': 'eigen', 'expression': '[[1,0],[0,1]]'})
    text = str(res).replace(' ', '')
    assert ok
    assert '代数重数为}2' in text and '几何重数:}2' in text, text
    assert '矩阵可对角化' in text and '矩阵不可对角化' not in text, text

    # A genuine Jordan block must still be reported as non-diagonalizable.
    ok, res = start_compute('eigen', {
        'operation_type': 'eigen', 'expression': '[[2,1],[0,2]]'})
    text = str(res).replace(' ', '')
    assert '代数重数为}2' in text and '几何重数:}1' in text, text
    assert '矩阵不可对角化' in text, text
