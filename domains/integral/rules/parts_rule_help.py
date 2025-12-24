from typing import Tuple
from sympy import Expr, Mul, Symbol, simplify

from utils import (
    is_exp, is_inv_trig, is_log, is_poly, is_trig
)

# ILAET rule for choosing 'u' in integration by parts:
# Lower number = higher priority for choosing 'u'
# Priority: Inverse Trig > Logarithmic > Algebraic > Exponential > Trigonometric
ILAET_PRIORITY = {
    'inverse_trig': 0,
    'log': 1,
    'algebraic': 2,
    'exp': 3,
    'trig': 4,
}


def _classify_factor(factor: Expr, var: Symbol) -> int:
    """Classify a factor according to ILAET rule for integration by parts.

    Assumption: 'factor' is a simple term (e.g., log(x), sin(x), x**2),

    Returns priority score (lower = better candidate for 'u').
    """
    # 1. Inverse Trigonometric
    if is_inv_trig(factor):
        return ILAET_PRIORITY['inverse_trig']

    # 2. Logarithmic: ln(x), log(x, b)
    if is_log(factor):
        return ILAET_PRIORITY['log']

    # 3. Algebraic: polynomial, rational, root expressions
    # Rough check: built from var using +, -, *, /, ** (rational exponents)
    if is_poly(factor, var):
        return ILAET_PRIORITY['algebraic']

    # 4. Exponential: exp(x), a**x (a constant)
    if is_exp(factor, var):
        return ILAET_PRIORITY['exp']

    # 5. Trigonometric
    if is_trig(factor):
        return ILAET_PRIORITY['trig']

    # Default: unknown function, treat as lowest priority
    return 5


def select_parts_u_dv(expr: Mul, var: Symbol) -> Tuple[Expr, Expr]:
    """Select u and dv for integration by parts using the ILAET heuristic."""
    factors = list(expr.args)
    # Initialize to a value higher than any ILAET priority
    best_priority = float('inf')
    u_candidate = None

    for factor in factors:
        # Skip pure constants (they go into dv)
        if factor.is_number or (factor.is_constant() and not factor.has(var)):
            continue
        priority = _classify_factor(factor, var)
        if priority < best_priority:
            best_priority = priority
            u_candidate = factor

    dv = simplify(expr / u_candidate)
    return u_candidate, dv
