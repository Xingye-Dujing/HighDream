import re
from functools import lru_cache

from sympy import Expr, evaluate, expand_log, latex, log, radsimp, simplify, sympify

from config import LARGE_POWER_EXPONENT_DIGITS

_HUGE_POWER_RE = re.compile(r'(?:\*\*|\^)\s*-?(\d+)')


def _has_huge_power_literal(expr_str: str) -> bool:
    """True if the input has a power whose integer exponent literal is very large.

    Detection uses the exponent literal's digit count as a cheap pre-check so we
    can avoid ever materializing huge integer powers such as ``2**1000000``.
    """
    return any(len(match.group(1)) >= LARGE_POWER_EXPONENT_DIGITS
               for match in _HUGE_POWER_RE.finditer(expr_str))


@lru_cache(maxsize=512)
def cached_latex(expr) -> str:
    return latex(expr)


@lru_cache(maxsize=256)
def cached_simplify(expr: Expr, inverse: bool = True) -> Expr:
    return simplify(expr, inverse=inverse)


@lru_cache(maxsize=1024)
def cached_doit(operation) -> Expr:
    return operation.doit()


@lru_cache(maxsize=256)
def cached_sympify(expr_str: str) -> Expr:
    # Callers may pass an already-sympified object (e.g. the limit calculator
    # pre-sympifies); sympify is then idempotent and the regex guard must not run.
    if not isinstance(expr_str, str):
        return sympify(expr_str)  # type: ignore
    if _has_huge_power_literal(expr_str):
        # Parse without evaluation so a huge integer power stays symbolic instead
        # of being computed (which wastes memory and breaks int -> str rendering).
        with evaluate(False):
            return sympify(expr_str)
    return sympify(expr_str)  # type: ignore


@lru_cache(maxsize=128)
def cached_simplify_radsimp(expr: Expr, is_radsimp: bool) -> Expr:
    if is_radsimp:
        expr = radsimp(expr)
    if expr.has(log):
        expr = expand_log(expr, force=True)
    return expr
