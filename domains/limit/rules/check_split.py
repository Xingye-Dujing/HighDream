
from sympy import (
    Add, Expr,  Mul, Pow, S, Symbol, exp, limit, log, oo, simplify, sin
)

from domains.limit import (
    check_combination_indeterminate, check_function_tends_to_zero,
    check_limit_exists, check_limit_exists_oo,
)


def check_mul_split(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Determines whether a multiplicative expression can be safely split into two parts
    such that:
      - The limit of each part exists (finite or infinite),
      - Their product does not yield an indeterminate form (e.g., 0*oo),
      - or at least one part matches a known standard limit pattern.

    This function first attempts to detect special forms that admit closed-form limits
    (e.g., sin(f(x)), log(1+f(x)), (exp(f(x)) - 1), or(1+f(x))**g(x) with f(x) to 0),
    which are commonly used in asymptotic analysis. If no such pattern is found,
    it falls back to a general multiplicative decomposition strategy.
    """
    if not isinstance(expr, Mul):
        return False
    factors = expr.as_ordered_factors()

    # A for loop to detect.
    for i, factor in enumerate(factors):
        # Strategy 1: Detect standard limit forms
        num, den = factor.as_numer_denom()
        # Detect sin(f(x)), f(x) to 0
        for part in (num, den):
            if isinstance(part, sin) and part.has(var) and check_function_tends_to_zero(part.args[0], var, point, direction):
                return True

        # Detect ln(1+f(x)), f(x) to 0
        for part in (num, den):
            if isinstance(part, log):
                f = part.args[0] - 1
                if f.has(var) and check_function_tends_to_zero(f, var, point, direction):
                    return True

        # Detect (exp(f(x))-1), f(x) to 0
        for part in (num, den):
            if isinstance(part, Add) and len(part.args) == 2:
                # SymPy orders terms so constant comes first
                if part.args[0] != -1:
                    continue
                if isinstance(part.args[1], exp):
                    f = part.args[1].args[0]
                    if f.has(var) and check_function_tends_to_zero(f, var, point, direction):
                        return True

        # Detect (f(x) + 1)**h(x), f(x) -> 0
        if isinstance(factor, Pow):
            base, _exp = factor.as_base_exp()
            inv_f = base - 1
            if check_function_tends_to_zero(inv_f, var, point, direction):
                # Check if f(x)*g(x) tends to a constant
                ratio = simplify(inv_f * _exp)
                if not ratio.has(var):
                    return True

        # Strategy 2: General multiplicative splitting
        first_part = factor
        rest_factors = factors[:i] + factors[i+1:]
        if not rest_factors:
            continue
        rest_part = Mul(*rest_factors)

        # Only split if both sub-limits exist and their combination is determinate
        if (check_limit_exists(first_part, var, point, direction) and
            check_limit_exists(rest_part, var, point, direction) and
                not check_combination_indeterminate(first_part, rest_part, var, point, direction, 'mul')):
            return True

    return False


def check_add_split(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Determines whether an additive expression can be split into two parts such that:
      - The limit of each part exists (finite or infinite),
      - Their sum does not yield an indeterminate form (e.g., oo-oo).

    The function iteratively isolates each term and checks whether both the term
    and the remainder of the sum admit well-defined limits whose combination is
    determinate under addition.
    """
    if not isinstance(expr, Add):
        return False
    terms = expr.as_ordered_terms()
    for i, term in enumerate(terms):
        # Detect the situation of extracting the i-th term
        first_part = term
        rest_terms = terms[:i] + terms[i+1:]
        rest_part = Add(*rest_terms) if rest_terms else S.Zero

        # Only split if both sub-limits exist and their combination is determinate
        if (check_limit_exists(first_part, var, point, direction) and
            check_limit_exists(rest_part, var, point, direction) and
                not check_combination_indeterminate(first_part, rest_part, var, point, direction, 'add')):
            return True
    return False


def check_div_split(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Determines whether a division expression can be safely evaluated by separately
    taking limits of numerator and denominator, i.e., when:

      1. The numerator has a well-defined limit (finite or infinite),
      2. The denominator has a well-defined, nonzero limit,
      3. The quotient does not correspond to an indeterminate form
         (e.g., 0/0 or oo/oo).
    """
    # Only consider expressions that are effectively divisions.
    # While as_numer_denom() works on any Expr, we require a nontrivial denominator.
    num, den = expr.as_numer_denom()
    if den == S.One:
        return False

    # Compute limits once to avoid redundant calls.
    try:
        num_limit = simplify(limit(num, var, point, dir=direction))
        den_limit = simplify(limit(den, var, point, dir=direction))
    except Exception:
        return False

    # Check that both limits exist in the extended real sense (finite or +-oo).
    if not check_limit_exists_oo(num_limit):
        return False
    if not check_limit_exists_oo(den_limit):
        return False

    # Denominator must not tend to zero.
    if den_limit == 0:
        return False

    # Reject classical indeterminate forms.
    if (num_limit == 0 and den_limit == 0) or \
       (num_limit in (oo, -oo) and den_limit in (oo, -oo)):
        return False

    return True
