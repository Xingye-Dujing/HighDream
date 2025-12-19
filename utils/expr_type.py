"""Expression Type Checking Utilities

This module provides utility functions to classify and identify types of mathematical
expressions using SymPy expressions. It includes checks for logarithmic, polynomial,
trigonometric, inverse trigonometric, exponential, and radical expressions.

Functions:
    is_log: Check if expression is a logarithmic function
    is_poly: Check if expression is a polynomial
    is_trig: Check if expression contains trigonometric functions
    is_inv_trig: Check if expression is an inverse trigonometric function
    is_exp: Check if expression is exponential
    has_radical: Check if expression contains radicals
"""

from sympy import (
    Expr, Pow, Symbol, asin, acos, atan,
    cos, cot, csc, exp, log, sec, sin, tan
)


def is_log(f: Expr) -> bool:
    """Determine if the given expression is a logarithmic function.

    Args:
        f (Expr): A SymPy expression to check

    Returns:
        bool: True if the expression is a logarithmic function, False otherwise

    Example:
        >>> from sympy import log, symbols
        >>> x = symbols('x')
        >>> is_log(log(x))
        True
        >>> is_log(x**2)
        False
    """
    return isinstance(f, log)


def is_poly(f: Expr, var: Symbol) -> bool:
    """Determine if the given expression is a polynomial with respect to a variable.

    Args:
        f (Expr): A SymPy expression to check
        var (Symbol): The variable to check polynomial status against

    Returns:
        bool: True if the expression is a polynomial in the given variable, False otherwise

    Example:
        >>> from sympy import symbols
        >>> x = symbols('x')
        >>> is_poly(x**2 + 3*x + 1, x)
        True
        >>> is_poly(1/x, x)
        False
    """
    return f.is_polynomial(var)


def is_trig(f: Expr) -> bool:
    """Determine if the given expression contains trigonometric functions.

    Checks for the presence of sin, cos, tan, sec, csc, or cot functions.

    Args:
        f (Expr): A SymPy expression to check

    Returns:
        bool: True if the expression contains trigonometric functions, False otherwise

    Example:
        >>> from sympy import sin, cos, symbols
        >>> x = symbols('x')
        >>> is_trig(sin(x) + cos(x))
        True
        >>> is_trig(x**2)
        False
    """
    return f.has(sin, cos, tan, sec, csc, cot)


def is_inv_trig(f: Expr) -> bool:
    """Determine if the given expression is an inverse trigonometric function.

    Specifically checks for asin, acos, or atan functions.

    Args:
        f (Expr): A SymPy expression to check

    Returns:
        bool: True if the expression is an inverse trigonometric function, False otherwise

    Example:
        >>> from sympy import asin, symbols
        >>> x = symbols('x')
        >>> is_inv_trig(asin(x))
        True
        >>> is_inv_trig(sin(x))
        False
    """
    return isinstance(f, (asin, acos, atan))


def is_exp(f: Expr, var: Symbol) -> bool:
    """Determine if the given expression is exponential with respect to a variable.

    An expression is considered exponential if it's either:
    1. The exp() function
    2. A power where base is numeric and exponent contains the variable

    Args:
        f (Expr): A SymPy expression to check
        var (Symbol): The variable to check exponential status against

    Returns:
        bool: True if the expression is exponential in the given variable, False otherwise

    Example:
        >>> from sympy import exp, symbols
        >>> x = symbols('x')
        >>> is_exp(exp(x), x)
        True
        >>> is_exp(2**x, x)
        True
        >>> is_exp(x**2, x)
        False
    """
    return isinstance(f, exp) or (isinstance(f, Pow) and f.base.is_number and f.exp.has(var))


def has_radical(f: Expr, var: Symbol) -> bool:
    """Determine if the given expression contains radicals (fractional powers) involving a variable.

    A radical is identified as a power expression with a rational exponent less than 1
    that contains the specified variable in its base.

    Args:
        f (Expr): A SymPy expression to check
        var (Symbol): The variable to check for in radicals

    Returns:
        bool: True if the expression contains radicals with the given variable, False otherwise

    Example:
        >>> from sympy import sqrt, symbols
        >>> x = symbols('x')
        >>> has_radical(sqrt(x), x)
        True
        >>> has_radical(x**(1/3), x)
        True
        >>> has_radical(x**2, x)
        False
    """
    if isinstance(f, Pow):
        return f.exp.is_Rational and f.exp < 1 and f.base.has(var)
    return any(has_radical(arg, var) for arg in f.args) if f.args else False
