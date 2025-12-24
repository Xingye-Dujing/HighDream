from sympy import (
    Expr, Symbol, log, exp, sin, cos, tan, cot, csc, sec,
    asin, acos, atan, acot, acsc, asec, sinh, cosh, tanh, coth,
    csch, sech, asinh, acosh, atanh, acoth, acsch, asech, sqrt,
    Add, Mul, Pow, Abs, floor, ceiling
)


def is_elementary_expression(expr: Expr) -> bool:
    """Check if an expression is elementary."""

    # Base cases
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
