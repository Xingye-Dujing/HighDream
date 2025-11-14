from typing import Dict, List, Tuple, Union
from sympy import (
    Expr, Interval, Intersection, Limit, Pow, S, Symbol,
    acos, asin, limit, log, oo, solveset, zoo
)
from sympy.core.relational import Relational


def detect_feasible_directions(expr: Expr, var: Symbol, point: Expr) -> Dict[str, bool]:
    """Determine whether left/right limits are potentially computable based on domain constraints.

    This function analyzes the expression's subcomponents (logarithms, even roots, inverse trigonometric
    functions, etc.) to infer whether the expression is defined in arbitrarily small neighborhoods
    to the left (var to point⁻) or right (var to point+) of the limit point.

    It does not guarantee the limit exists—only that the expression is potentially defined
    near the point from that side.

    Args:
        expr (Expr): The symbolic expression to analyze.
        var (Symbol): The variable with respect to which the limit is taken.
        point (Expr): The limit point (e.g., 0, oo).

    Returns:
        Dict[str, bool]: A dictionary with keys 'left' and 'right'.
        - True: The expression may be defined in some punctured neighborhood on that side.
        - False: The expression is provably undefined arbitrarily close on that side.

    Note:
        - For infinite points (+-oo), both directions are conservatively marked as feasible,
          since directional analysis at infinity is context-dependent and often unnecessary.
        - Uses purely symbolic methods—no floating-point perturbations (e.g., point +- 1e-6).
        - If constraint solving fails (e.g., transcendental inequalities), assumes feasibility
          (conservative "fail-open" policy to avoid false negatives).
    """

    # Handle infinite limit points
    if point in (oo, -oo):
        return {'left': True, 'right': True}

    # Early exit: if point is not real, directional limits are ill-defined in real calculus
    if not point.is_real:
        return {'left': False, 'right': False}

    # Collect domain constraints as relational expressions (e.g., f(var) > 0)
    constraints: List[Relational] = []

    # 1. Logarithms: log(u) requires u > 0
    for log_expr in expr.atoms(log):
        arg = log_expr.args[0]
        constraints.append(arg > 0)

    # 2. Even roots: u**(p/q) with q even and p/q > 0 requires u >= 0
    for pow_expr in expr.atoms(Pow):
        base, exp = pow_expr.args
        if exp.is_Rational and exp > 0 and exp.q % 2 == 0:
            constraints.append(base >= 0)

    # 3. Inverse sine/cosine: asin(u), acos(u) require -1 <= u <= 1
    for inv_trig in expr.atoms(asin, acos):
        arg = inv_trig.args[0]
        constraints.append(arg >= -1)
        constraints.append(arg <= 1)

    # If no domain-restricting subexpressions, all directions are feasible
    if not constraints:
        return {'left': True, 'right': True}

    # Define left and right punctured neighborhoods symbolically
    # Use open intervals approaching the point
    left_domain = Interval.open(-oo, point)
    right_domain = Interval.open(point, oo)

    feasible = {'left': True, 'right': True}

    for constr in constraints:
        # Skip non-relational or malformed constraints
        if not isinstance(constr, Relational):
            continue

        # Determine where the constraint holds
        try:
            solution_set = solveset(constr, var, domain=S.Reals)
        except (NotImplementedError, ValueError):
            # Cannot solve symbolically to assume constraint can be satisfied nearby
            continue

        # If solution set is empty globally, mark both directions as infeasible
        if solution_set.is_empty:
            feasible['left'] = feasible['right'] = False
            break

        # Intersect with directional domains to check local feasibility
        try:
            left_intersect = Intersection(solution_set, left_domain)
            right_intersect = Intersection(solution_set, right_domain)
        except Exception:
            # Fallback: assume feasible if intersection fails
            continue

        # A direction is infeasible only if the intersection is provably empty
        if hasattr(left_intersect, 'is_empty') and left_intersect.is_empty:
            feasible['left'] = False
        if hasattr(right_intersect, 'is_empty') and right_intersect.is_empty:
            feasible['right'] = False

        # Early termination if both directions are ruled out
        if not feasible['left'] and not feasible['right']:
            break

    return feasible


def get_limit_args(context: Dict[str, Union[Symbol, Expr, str]]) -> Tuple[Symbol, Expr, str]:
    """Extract limit parameters from the evaluation context.

    Retrieves the limit variable, point, and direction from a context dictionary.
    This utility ensures consistent access to limit metadata across matchers and solvers.
    """
    var, point = context['variable'], context['point']
    direction = context.get('direction', '+')
    return var, point, direction


def check_limit_exists(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Check whether a directional limit exists and is finite.

    Attempts to evaluate the limit of expr as var approaches point from the specified
    direction. Returns True only if the limit evaluates to a finite real (or complex) value.
    Infinite limits (oo, -oo), undefined results (Nan), oscillatory behavior, or evaluation
    failures are all treated as non-existent for the purpose of this function.
    """
    try:
        lim_val = Limit(expr, var, point, dir=direction).doit()
        return lim_val.is_finite
    except Exception:
        return False


def check_limit_exists_oo(lim_val: Expr) -> bool:
    """Determine whether a limit value is considered to exis" in extended real analysis.

    In the context of real-variable calculus, a limit is often said to exist if it
    evaluates to a finite real number or diverges to positive/negative infinity.
    This function returns True for such cases, and False for indeterminate forms,
    undefined expressions (e.g., NaN), or complex infinities.
    """
    if lim_val.is_finite:
        return True
    if lim_val in (S.Infinity, S.NegativeInfinity):
        return True
    return False


def is_infinite(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Check whether the directional limit of an expression diverges to infinity.

    Computes the limit of expr as var approaches point from the specified direction.
    Returns True if the result is positive or negative real infinity (oo or -oo).
    """
    try:
        lim_val = limit(expr, var, point, dir=direction)
        return lim_val in (S.Infinity, S.NegativeInfinity)
    except Exception:
        return False


def is_zero(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Check whether the directional limit of an expression equals zero.

    Computes the limit of expr as var approaches point from the specified direction.
    Returns True if the result is exactly zero in the symbolic sense.
    """

    try:
        lim_val = limit(expr, var, point, dir=direction)
        return lim_val == S.Zero
    except Exception:
        return False


def is_constant(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Check whether the directional limit of an expression evaluates to a finite numeric constant.

    A constant in this context means a concrete, finite number—such as an integer,
    rational, floating-point number, or symbolic constant like pi or E—that does not
    depend on free symbols and is not infinite (oo, -oo, zoo) or indeterminate (NaN).

    This function returns True if the limit exists and is a finite, symbol-free number.
    Expressions like sin(pi/2) (which evaluates to 1) are accepted; expressions like x,
    a+1 (with free symbol a), or oo are not.
    """

    try:
        lim = limit(expr, var, point, dir=direction)
        return lim.is_real and not lim.has(oo, -oo, zoo)
    except Exception:
        return False


def check_function_tends_to_zero(f: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Check whether a function tends to zero in a given directional limit.

    Attempts to compute the one-sided or two-sided limit of f as var approaches point
    from the specified direction. Returns True only if the limit exists and is exactly zero.
    """
    try:
        lim_val = limit(f, var, point, dir=direction)
        return lim_val == S.Zero
    except Exception:
        return False


def is_indeterminate_form(expr: Expr, var: Symbol, point: Expr, direction: str) -> bool:
    """Determine whether an expression exhibits an indeterminate form at a limit point.

    This function analyzes the structure and limit behavior of expr as var to point
    from the given direction to detect classical indeterminate forms:

    - 0/0, oo/oo
    - 0*oo
    - oo-oo
    - 1^oo, 0^0, oo^0

    Note that this is a heuristic structural check, not a definitive mathematical test.
    It evaluates subexpression limits to classify the form. If any sub-limit fails to evaluate,
    the function conservatively assumes an indeterminate form may be present.

    Args:
        expr (Expr): The symbolic expression to analyze.
        var (Symbol): The variable approaching the limit point.
        point (Expr): The point being approached (e.g., 0, oo).
        direction (str): Direction of approach ('+' or '-').

    Returns:
        bool: True if an indeterminate form is detected; False otherwise.
              Returns True on evaluation failure (conservative fallback).

    Examples:
        >>> x = symbols('x')
        >>> is_indeterminate_form(sin(x)/x, x, 0, '+')      # 0/0
        True
        >>> is_indeterminate_form(x*log(x), x, 0, '+')     # 0*(-oo)
        True
        >>> is_indeterminate_form(exp(x)/x, x, oo, '+')      # oo/oo
        True
        >>> is_indeterminate_form(x+1, x, 0, '+')          # regular
        False
    """
    try:
        # Helper to safely compute sub-limits
        def _limit(e: Expr) -> Expr:
            return limit(e, var, point, dir=direction)

        # Case 1: a/b to check 0/0 or oo/oo
        if expr.is_Mul:
            numer, denom = expr.as_numer_denom()
            if denom != 1:  # Actually a division
                try:
                    L_num = _limit(numer)
                    L_den = _limit(denom)
                except Exception:
                    return True  # Conservative: assume indeterminate

                # Check 0/0
                if L_num == S.Zero and L_den == S.Zero:
                    return True
                # Check oo/oo (including -oo)
                if (L_num in (oo, -oo)) and (L_den in (oo, -oo)):
                    return True

        # Case 2: Multiplication to check 0*oo
        if expr.is_Mul:
            factors = expr.as_ordered_factors()
            has_zero = False
            has_inf = False
            for f in factors:
                try:
                    L_f = _limit(f)
                except Exception:
                    return True
                if L_f == S.Zero:
                    has_zero = True
                elif L_f in (oo, -oo):
                    has_inf = True
            if has_zero and has_inf:
                return True

        # Case 3: Addition to check oo-oo
        if expr.is_Add:
            terms = expr.as_ordered_terms()
            has_pos_inf = False
            has_neg_inf = False
            for t in terms:
                try:
                    L_t = _limit(t)
                except Exception:
                    return True
                if L_t == oo:
                    has_pos_inf = True
                elif L_t == -oo:
                    has_neg_inf = True
            if has_pos_inf and has_neg_inf:
                return True

        # Case 4: Power to check 1^oo, 0^0, oo^0
        if expr.is_Pow:
            base, exp = expr.base, expr.exp
            try:
                L_base = _limit(base)
                L_exp = _limit(exp)
            except Exception:
                return True

            # 1^oo
            if L_base == S.One and L_exp in (oo, -oo):
                return True
            # 0^0
            if L_base == S.Zero and L_exp == S.Zero:
                return True
            # oo^0
            if L_base in (oo, -oo) and L_exp == S.Zero:
                return True

        return False

    except Exception:
        # Top-level fallback: if anything goes wrong, assume indeterminate
        return True


def check_combination_indeterminate(part1: Expr, part2: Expr, var: Symbol, point: Expr, direction: str, operation: str) -> bool:
    """Check whether combining two subexpressions yields an indeterminate form.

    This function evaluates the limits of part1 and part2 as var to point from the
    specified direction and checks for classical indeterminate combinations:

    - For operation='mul': detects 0*oo or oo*0 (including sign variants like 0*(-oo)).
    - For operation='add': detects oo-oo (i.e., oo+(-oo) or -oo+oo).

    Args:
        part1 (Expr): First subexpression.
        part2 (Expr): Second subexpression.
        var (Symbol): Limit variable.
        point (Expr): Point being approached (e.g., 0, oo).
        direction (str): Direction of approach; typically '+' or '-'.
        operation (str): Binary operation to simulate. Must be either:
            - 'mul' for multiplication,
            - 'add' for addition.

    Returns:
        bool: True if the combination results in an indeterminate form;
              False otherwise (including when limits are finite or determinate).

    Raises:
        ValueError: If operation is not 'mul' or 'add'.

    Note:
        This function assumes real-valued limits. Complex infinities (zoo) are not treated
        as indeterminate in this context. Evaluation failures (e.g., non-convergent sub-limits)
        result in False—the function does not conservatively assume indeterminacy on error,
        unlike broader form detectors.

    Examples:
        >>> x = symbols('x')
        >>> check_combination_indeterminate(x, 1/x, x, 0, '+', 'mul')   # 0*oo
        True
        >>> check_combination_indeterminate(1/x, -1/x, x, 0, '+', 'add')  # oo+(-oo)
        True
        >>> check_combination_indeterminate(x, x, x, 0, '+', 'mul')       # 0*0
        False
    """
    if operation not in ('mul', 'add'):
        raise ValueError(
            f"Unsupported operation: {operation!r}. Expected 'mul' or 'add'.")

    try:
        lim1 = limit(part1, var, point, dir=direction)
        lim2 = limit(part2, var, point, dir=direction)
    except Exception:
        # If either sub-limit fails to evaluate, we cannot confirm indeterminacy.
        # Unlike full-expression analyzers, this helper returns False on error.
        return False

    if operation == 'mul':
        # Check for 0 * (±oo) or (±oo) * 0
        is_zero1 = lim1 == S.Zero
        is_zero2 = lim2 == S.Zero
        is_inf1 = lim1 in (oo, -oo)
        is_inf2 = lim2 in (oo, -oo)

        return (is_zero1 and is_inf2) or (is_inf1 and is_zero2)

    if operation == 'add':
        # Check for oo + (-oo) or (-oo) + oo
        return (lim1 == oo and lim2 == -oo) or (lim1 == -oo and lim2 == oo)

    # Unreachable due to earlier validation
    return False
