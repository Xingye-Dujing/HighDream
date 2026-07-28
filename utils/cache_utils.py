from functools import lru_cache

from sympy import Expr, expand_log, latex, log, radsimp, simplify, sympify


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
    return sympify(expr_str)  # type: ignore


@lru_cache(maxsize=128)
def cached_simplify_radsimp(expr: Expr, is_radsimp: bool) -> Expr:
    if is_radsimp:
        expr = radsimp(expr)
    if expr.has(log):
        expr = expand_log(expr, force=True)
    return expr
