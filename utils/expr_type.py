from sympy import (
    Expr, Pow, Symbol, asin, acos, atan,
    cos, cot, csc, exp, log, sec, sin, tan
)


def is_log(f: Expr) -> bool:
    return isinstance(f, log)


def is_poly(f: Expr, var: Symbol) -> bool:
    return f.is_polynomial(var)


def is_trig(f: Expr) -> bool:
    return f.has(sin, cos, tan, sec, csc, cot)


def is_inv_trig(f: Expr) -> bool:
    return isinstance(f, (asin, acos, atan))


def is_exp(f: Expr, var: Symbol) -> bool:
    return isinstance(f, exp) or (isinstance(f, Pow) and f.base.is_number and f.exp.has(var))


def has_radical(f: Expr, var: Symbol) -> bool:
    if isinstance(f, Pow):
        return f.exp.is_Rational and f.exp < 1 and f.base.has(var)
    return any(has_radical(arg, var) for arg in f.args) if f.args else False
