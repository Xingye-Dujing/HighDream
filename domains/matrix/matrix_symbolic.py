"""Symbolic matrix expression simplification (uppercase symbols = n x n matrices).

Reduces MatExpr using the standard linear-algebra identities that undergrad
courses expect, recording one step per transformation so the UI can show work:

    A*I = A,  A*A^-1 = I,  (A^-1)^-1 = A,  (A*B)^-1 = B^-1*A^-1,
    (A*B)^T = B^T*A^T,  A*(B + C) = A*B + A*C, ...

All free identifiers are treated as square matrices of a shared symbolic size
``n`` so products are dimension-valid.
"""

import re
from typing import List, Tuple

from sympy import Expr, Symbol, expand
from sympy.matrices.expressions import Identity, Inverse, MatMul, MatrixSymbol, Transpose
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

from utils.cache_utils import _has_huge_power_literal, _reject_unsafe_syntax

_N = Symbol('n')
_TOKEN_RE = re.compile(r'[A-Za-z_]\w*')
_RESERVED = {'Inverse', 'Transpose', 'Identity', 'MatMul', 'MatAdd', 'n'}


def _free_names(text: str) -> set:
    """Identifier tokens that are matrix names (not built-in functions)."""
    return {tok for tok in _TOKEN_RE.findall(text) if tok not in _RESERVED}


def _parse_matrix_expr(text: str) -> Expr:
    """Parse ``text`` mapping every free identifier to an n x n MatrixSymbol."""
    local: dict = {name: MatrixSymbol(name, _N, _N) for name in _free_names(text)}
    local['n'] = _N
    local['Inverse'] = Inverse
    local['Transpose'] = Transpose
    local['Identity'] = Identity
    return parse_expr(text, local_dict=local,
                      transformations=standard_transformations)


def _is_inverse_target(e) -> bool:
    return isinstance(e, Inverse) and isinstance(e.arg, (MatMul, Inverse))


def _flip_inverse(e):
    """Rewrite (A*B)^-1 -> B^-1*A^-1 and (A^-1)^-1 -> A."""
    inner = e.arg
    if isinstance(inner, MatMul):
        return MatMul(*[Inverse(f) for f in reversed(inner.args)])
    if isinstance(inner, Inverse):
        return inner.arg
    return e


def simplify_matrix_symbolic(text: str) -> List[Tuple[Expr, str]]:
    """Return ``[(expr, reason), ...]`` — a step-by-step symbolic simplification.

    Raises on invalid input; the caller turns exceptions into an error result.
    """
    raw = str(text or '').strip()
    if not raw:
        raise ValueError('请输入符号矩阵表达式，例如 (A+B)*(A-B)')
    if '[' in raw or '\\begin' in raw:
        raise ValueError(
            '符号矩阵化简处理大写矩阵名（A、B、…）组成的表达式；'
            '具体数值矩阵请使用矩阵实验中的 rref/transpose/inverse 等操作'
        )
    if not _free_names(raw):
        raise ValueError('表达式中没有矩阵名；请使用 A、B 等大写字母表示矩阵')
    # parse_expr uses eval: an attribute chain such as
    # ``().__class__.__base__.__subclasses__()`` is self-contained (never
    # references the local dict), so binding free names to MatrixSymbol does
    # NOT sandbox it. Block attribute access / dunder names / calls to
    # non-symbols outright, exactly as the scalar parser does.
    _reject_unsafe_syntax(raw)
    # A huge power literal (2**1000000 style) is not something a symbolic
    # matrix simplifier should materialize: decline instead of hanging the worker.
    if _has_huge_power_literal(raw):
        raise ValueError('表达式包含过大的幂字面量，请避免使用超大指数')
    expr = _parse_matrix_expr(raw)
    steps: List[Tuple[Expr, str]] = [(expr, '原始符号矩阵表达式')]

    # 1) Apply inverse/identity rewrites, then collapse via doit (A*A^-1 -> I etc.).
    flipped = expr
    for _ in range(6):
        nxt = flipped.replace(_is_inverse_target, _flip_inverse).doit()
        if nxt == flipped:
            break
        flipped = nxt
    if flipped != expr:
        steps.append((flipped, '逆与单位矩阵化简: A·A⁻¹=I, A·I=A, (A⁻¹)⁻¹=A, (AB)⁻¹=B⁻¹A⁻¹'))

    # 2) Distribute matrix products over sums.
    distributed = expand(flipped, matrix=True).doit()
    if distributed != flipped:
        steps.append((distributed, '乘法对加法的分配律展开'))

    # 3) Final collapse after distribution.
    final = distributed.doit()
    if final != distributed:
        steps.append((final, '合并同类项并再次化简'))

    return steps
