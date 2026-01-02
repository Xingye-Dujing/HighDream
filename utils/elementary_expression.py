from sympy import (
    Expr, I, Symbol, log, exp, sin, cos, tan, cot, csc, sec,
    asin, acos, atan, acot, acsc, asec, sinh, cosh, tanh, coth,
    csch, sech, asinh, acosh, atanh, acoth, acsch, asech, sqrt,
    Add, Mul, Pow, Abs, floor, ceiling
)


def is_elementary_expression(expr: Expr) -> bool:
    """Check if an expression is elementary."""
    if not expr:
        return False

    # Base cases
    if expr.has(I):
        return False
    if expr.is_number or isinstance(expr, Symbol):
        return True

    elementary_types = (
        Add, Mul, Pow, sqrt, Abs,  # Basic arithmetic
        log, exp,  # Exponential and logarithmic
        sin, cos, tan, cot, csc, sec,  # Trigonometric
        asin, acos, atan, acot, acsc, asec,  # Inverse trigonometric
        sinh, cosh, tanh, coth, csch, sech,  # Hyperbolic
        asinh, acosh, atanh, acoth, acsch, asech,  # Inverse hyperbolic
        floor, ceiling  # Rounding functions
    )

    # Check if the expression's class is in the elementary types
    if type(expr) in elementary_types:
        # Recursively check all arguments
        for arg in expr.args:
            if not is_elementary_expression(arg):
                return False
        return True

    return False


def can_use_weierstrass(expr: Expr, var: Symbol) -> bool:
    """Determine whether expr is a pure trigonometric rational expression of the form
    R(sin(x), cos(x), tan(x), ...), i.e., whether Weierstrass substitution can be used.

    Requirements:
    - Contains only sin(x), cos(x), tan(x), cot(x), sec(x), csc(x)
    - Does not contain bare x
    - Does not contain non-trigonometric functions such as exp, log, sqrt(x), asin, etc.
    - The exponent in power operations must be integers (to ensure it's a rational expression)
    """
    if expr is None:
        return False

    # Base cases
    if expr.is_constant():
        return True

    if isinstance(expr, Symbol):
        return False

    allowed_trig = {sin, cos, tan, cot, sec, csc}

    expr_type = type(expr)

    if expr_type in (Add, Mul):
        # Recursively check all sub-expressions
        return all(can_use_weierstrass(arg, var) for arg in expr.args)

    if expr_type == Pow:
        base, _exp = expr.args
        return can_use_weierstrass(base, var) and _exp.is_number

    if expr_type in allowed_trig:
        if len(expr.args) != 1:
            return False
        arg = expr.args[0]
        if arg == var:
            return True
        return False

    # Any other types (such as exp, log, asin, sqrt, Derivative, etc.) are invalid
    return False
