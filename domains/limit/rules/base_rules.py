from sympy import (
    AccumBounds, Add, Expr, Integer, Limit, Mul, Pow, S, UnevaluatedExpr,
    exp, latex, limit, log, nan, oo, simplify, sin, sqrt, zoo
)

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn
from domains.limit import (
    check_add_split, check_combination_indeterminate, check_div_split,
    check_function_tends_to_zero, check_limit_exists, check_mul_split,
    get_limit_args, is_constant, is_indeterminate_form, is_infinite, is_zero,
)


def direct_substitution_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the direct substitution rule for evaluating a limit."""
    var, point, direction = get_limit_args(context)

    result = Limit(expr, var, point, dir=direction)
    # Determine whether to skip showing the intermediate substitution step
    skip_intermediate = (
        expr == var
        or expr.is_number
        or result.is_infinite
    )
    lhs, rhs = latex(result), latex(result.doit())

    if skip_intermediate:
        full_rule = f"{lhs} = {rhs}"
    else:
        # Perform substitution without evaluation to display the intermediate form.
        expr_subbed = expr.subs(var, UnevaluatedExpr(point))
        full_rule = f"{lhs} = {latex(expr_subbed)} = {rhs}"

    return result.doit(), f"直接代入: ${full_rule}$"


def mul_split_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply multiplicative splitting rules to decompose a limit expression.

    This function attempts two strategies in order:

    1. Standard limit form extraction: Identify and extract well-known
       indeterminate forms that converge to standard limits (e.g.,
       sin(f(x))/f(x) to 1 as f(x) to 0, ln(1+f(x))/f(x) to 1, etc.).

    2. General multiplicative splitting: If no standard form is found,
       split the expression into two parts whose individual limits exist
       and whose product does not yield an indeterminate form.
    """

    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)
    factors = expr.as_ordered_factors()

    # Two for loops to apply the two strategies successively.
    # Strategy 1: Extract standard limit forms
    for i, factor in enumerate(factors):
        new_factor = None
        num, den = factor.as_numer_denom()

        # Case: sin(f(x))/f(x) or f(x)/sin(f(x)), where f(x) to 0
        if isinstance(num, sin) and num.has(var) and check_function_tends_to_zero(num, var, point, direction):
            new_factor = factor.args[0]  # f(x)
        elif isinstance(den, sin) and den.has(var) and check_function_tends_to_zero(den, var, point, direction):
            new_factor = 1/factor.args[0].args[0]  # 1/f(x)

        # Case: ln(1+f(x))/f(x) or f(x)/ln(1 + f(x)), where f(x) to 0
        elif isinstance(num, log):
            if factor.has(var) and check_function_tends_to_zero(factor, var, point, direction):
                new_factor = num.args[0] - 1  # f(x)
        elif isinstance(den, log):
            if factor.has(var) and check_function_tends_to_zero(factor, var, point, direction):
                new_factor = 1/(den.args[0] - 1)  # 1/(f(x))

        # Case: (exp(f(x))-1)/f(x) or f(x)/(exp(f(x))-1), where f(x) to 0
        elif isinstance(num, Add) and len(num.args) == 2:
            other = num.args[0]
            if other != -1:
                continue
            f = num.args[1].args[0]
            if not f.has(var) and check_function_tends_to_zero(f, var, point, direction) and not isinstance(num.args[1], exp):
                continue
            new_factor = f
        elif isinstance(den, Add) and len(den.args) == 2:
            other = den.args[0]
            if other != -1:
                continue
            f = den.args[1].args[0]
            if not f.has(var) and check_function_tends_to_zero(f, var, point, direction) or not isinstance(den.args[1], exp):
                continue
            new_factor = 1/f

        # Case: (1+f(x))**h(x) with f(x) to 0 and f(x)*h(x) to constant
        elif isinstance(factor, Pow):
            base, _exp = factor.as_base_exp()
            inv_f = base - 1
            if check_function_tends_to_zero(inv_f, var, point, direction):
                ratio = simplify(inv_f * _exp)
                if not ratio.has(var):  # limit of f*h exists and is constant
                    new_factor = 1  # entire factor is a standard exponential limit

        # Apply transformation if a standard form was matched
        if new_factor is not None:
            rest_factors = factors[:i] + [new_factor] + factors[i+1:]
            rest_part = Mul(*rest_factors)

            new_expr = Limit(factor / new_factor, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            rule_text = f"{latex(expr_limit)} = {latex(new_expr)}"
            return new_expr, f"应用重要极限的乘法拆分规则: ${rule_text}$"

    # Strategy 2: General multiplicative splitting
    for i, factor in enumerate(factors):
        first_part = factor
        rest_factors = factors[:i] + factors[i+1:]
        if not rest_factors:
            continue
        rest_part = Mul(*rest_factors)

        # Only split if both sub-limits exist and their combination is determinate
        if (check_limit_exists(first_part, var, point, direction) and
            check_limit_exists(rest_part, var, point, direction) and
                not check_combination_indeterminate(first_part, rest_part, var, point, direction, 'mul')):

            new_expr = Limit(first_part, var, point, dir=direction) * \
                Limit(rest_part, var, point, dir=direction)

            return new_expr, f"应用乘法拆分规则: ${latex(expr_limit)} = {latex(new_expr)}$"

    return None


def add_split_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply additive splitting rule to decompose a limit of a sum.

    This rule attempts to split the expression into two parts such that:

    - The limit of each part exists individually, and
    - Their combination does not result in an indeterminate form (e.g., oo − oo).

    The function iteratively considers prefixes of the ordered terms
    (from one term up to all but the last) as the first part, and the
    remainder as the second part.
    """
    var, point, direction = get_limit_args(context)
    terms = expr.as_ordered_terms()
    n = len(terms)

    # Try all non-trivial prefix splits: [term_0], [term_0 + term_1], ..., up to all but last
    for i in range(n):
        first_part = Add(*terms[:i+1])
        rest_terms = terms[i+1:] if i+1 < n else []
        rest_part = Add(*rest_terms) if rest_terms else S.Zero

        # Only split if both sub-limits exist and their sum is determinate
        if (check_limit_exists(first_part, var, point, direction) and
            check_limit_exists(rest_part, var, point, direction) and
                not check_combination_indeterminate(first_part, rest_part, var, point, direction, 'add')):

            new_expr = Limit(first_part, var, point, dir=direction) + \
                Limit(rest_part, var, point, dir=direction)
            expr_limit = Limit(expr, var, point, dir=direction)

            return new_expr, f"应用加法拆分规则: ${latex(expr_limit)} = {latex(new_expr)}$"

    return None


def div_split_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the quotient rule for limits."""
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)

    num, den = expr.as_numer_denom()

    num_limit_expr = Limit(num, var, point, dir=direction)
    den_limit_expr = Limit(den, var, point, dir=direction)
    new_expr = num_limit_expr / den_limit_expr

    return new_expr, f"应用除法拆分规则: ${latex(expr_limit)} = {latex(new_expr)}$"


def const_inf_add_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the constant-plus-infinity rule for limits.

    This rule handles sums where:

    - At least one term tends to +-oo,
    - All infinite terms share the same sign (all +oo or all −oo),
    - All remaining (non-infinite) terms have finite limits (i.e., are bounded near the limit point).

    In such cases, the overall limit is determined solely by the infinite part.
    """
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)
    terms = expr.as_ordered_terms()

    inf_sign = None
    for term in terms:
        # Find the sign from the first infinite term
        lim_val = limit(term, var, point, dir=direction)
        if lim_val not in (oo, -oo):
            continue
        inf_sign = 1 if lim_val == oo else -1 if lim_val == -oo else None
        break

    result = oo * inf_sign

    explanation = rf"应用\,趋于常数(有界)+-趋于无穷\, 规则(所有无穷项同号): ${latex(expr_limit)} = {latex(result)}$"
    return result, explanation


def const_inf_div_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the bounded-over-infinity limit rule.

    This rule applies when:

    - The expression is a quotient (num/den),
    - The numerator has a finite limit (i.e., is bounded near the limit point),
    - The denominator tends to +-oo.

    In such cases, the overall limit is 0.
    """
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)

    return Integer(0), rf"应用\,趋于常数(有界)/趋于无穷\,规则: ${latex(expr_limit)} = 0$"


def const_inf_mul_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the non-zero-constant-times-infinity limit rule.

    This rule applies when the expression is a product such that:

    - At least one factor tends to +-oo,
    - All other factors tend to finite limits,
    - The product of the finite limits is non-zero.

    In such cases, the overall limit is +-oo.
    """
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)
    factors = expr.as_ordered_factors()

    count = 0
    for factor in factors:
        lim_val = limit(factor, var, point, dir=direction)
        if lim_val == -oo or lim_val < 0:
            count += 1

    total_sign = -1 if (count % 2) else 1
    result = oo * total_sign

    return result, rf"应用\,趋于非零常数(有界)(可无)$\cdot$趋于无穷\,规则: ${latex(expr_limit)} = {latex(result)}$"


def small_o_add_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the sum-of-infinitesimals rule.

    This rule applies when the expression is a finite sum (or difference)
    of terms, each of which tends to 0 as the variable approaches the limit point.

    In such cases, the overall limit is 0.
    """
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)

    return Integer(0), rf"应用\,多个趋于\,0\,相加减\,规则: ${latex(expr_limit)} = 0$"


def const_zero_div_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the non-zero-constant-over-zero limit rule.

    This rule applies when:
      - The expression is a quotient (num/den),
      - num to L != 0 (finite and non-zero),
      - den to 0 (from a specific side, so sign is determined).

    The result is +-oo, with sign = sign(L) * sign(den near point).
    """
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)
    num, _ = expr.as_numer_denom()

    sign = 1 if direction == '+' else -1
    # SymPy moves the minus sign to the numerator,
    # so we only need to consider the sign of the limit of the numerator.
    num_lim = limit(num, var, point, dir=direction).doit()
    sign *= 1 if num_lim > 0 else -1
    result = oo * sign

    return result, rf"应用\,趋于非零常数(有界)/趋于0\,规则(可能需要通分再观察): ${latex(expr_limit)} = {latex(result)}$"


def conjugate_rationalize_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Rationalize a fraction whose numerator is of the form sqrt(A) +- sqrt(B).

    This rule applies when:
      - The expression is a quotient,
      - The numerator is a sum/difference of exactly two terms,
      - Each term is a square root (possibly with a coefficient of +-1),
      - The denominator is non-zero near the limit point.

    It multiplies numerator and denominator by the conjugate to eliminate radicals.
    """
    var, point, direction = get_limit_args(context)
    expr_limit = Limit(expr, var, point, dir=direction)
    num, den = expr.as_numer_denom()

    # Construct conjugate: flip the sign between the two terms
    # Original: a + b  to conjugate = a - b
    # Original: a - b  to conjugate = a + b
    # Since num = term1 + term2 (SymPy always stores as Add), we use a - b as conjugate
    a, b = num.args
    conj = a - b
    new_num = simplify(a**2 - b**2)
    new_den = simplify(den * conj)
    new_expr = simplify(new_num / new_den)
    new_limit = Limit(new_expr, var, point, dir=direction)

    explanation = (
        rf"$分子含有根号差，乘以共轭\,{latex(conj)}\,进行有理化:"
        f"{latex(expr_limit)} ="
        f"\\lim_{{{var} \\to {latex(point)}{direction}}}"
        f"\\frac{{({latex(num)})({latex(conj)})}}{{{latex(den)}({latex(conj)})}}="
        f"{latex(new_limit)}"
    )

    return new_limit, explanation


def direct_substitution_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Determine whether the direct substitution rule is applicable for evaluating the limit.

    Direct substitution is valid if:

    - Substituting the limit point into the expression yields a well-defined value
      (finite, +-oo, or complex infinity zoo), and
    - The resulting form is not an indeterminate form (e.g., 0/0, oo/oo, 0*oo, oo-oo, 1^oo, 0^oo, oo^0).
    """
    var, point, direction = get_limit_args(context)
    try:
        # Perform naive substitution of the limit point
        substituted_value = expr.subs(var, point)

        # Complex infinity (zoo) is acceptable — e.g., 1/x as x yo 0
        if substituted_value.has(zoo):
            return 'direct_substitution'

        # Explicit NaN indicates an undefined or indeterminate result
        if substituted_value is nan:
            return None

        # Trivial cases: constant or the variable itself
        if expr.is_number or expr == var:
            return 'direct_substitution'

        # Check for indeterminate subexpressions in multiplicative factors
        factors = expr.as_ordered_factors()
        for factor in factors:
            if is_indeterminate_form(factor, var, point, direction):
                return None

        # Final fallback: compute the actual limit to verify existence
        lim_val = limit(expr, var, point, dir=direction)
        if lim_val.is_finite or lim_val in (oo, -oo):
            return 'direct_substitution'
        return None
    except Exception:
        return None


def mul_split_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Determine whether multiplicative splitting is applicable.

    This matcher checks if the expression is a product that can be safely split
    into two non-empty subexpressions such that:

    - The limit of each part exists (finite or infinite), and
    - Their combination does not result in an indeterminate form (e.g., 0*oo).
    """
    var, point, direction = get_limit_args(context)
    if check_mul_split(expr, var, point, direction):
        return 'mul_split'
    return None


def add_split_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Determine whether additive splitting is applicable.

    This matcher checks if the expression is a sum that can be safely split
    into two non-empty subexpressions such that:

    - The limit of each part exists (finite or infinite), and
    - Their combination does not yield an indeterminate form (e.g., oo−oo).
    """
    var, point, direction = get_limit_args(context)
    if check_add_split(expr, var, point, direction):
        return 'add_split'
    return None


def div_split_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Determine whether the quotient rule for limits is applicable.

    This matcher checks if the expression is a quotient (i.e., a rational expression)
    for which the limit can be evaluated as the quotient of the limits of its
    numerator and denominator, provided that:

    - The limits of both numerator and denominator exist (finite or infinite),
    - The limit of the denominator is non-zero, and
    - The resulting form is not indeterminate (e.g., 0/0 or oo/oo).
    """
    var, point, direction = get_limit_args(context)
    if check_div_split(expr, var, point, direction):
        return 'div_split'
    return None


def const_inf_add_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions of the form 'bounded_term +- unbounded_terms',
    where all unbounded terms tend to infinity with the same sign.

    This matcher identifies sums consisting of:
      - Zero or more terms that approach a finite real limit (i.e., bounded),
      - One or more terms that diverge to either +oo or -oo,
      - All divergent terms must share the same sign at the limit point.
    """
    if not isinstance(expr, Add):
        return None
    var, point, direction = get_limit_args(context)
    terms = expr.as_ordered_terms()

    have_inf = False
    inf_sign = None

    for term in terms:
        lim_val = limit(term, var, point, dir=direction)
        if lim_val in (oo, -oo):
            have_inf = True
            term_sign = 1 if lim_val == oo else -1
            if inf_sign is None:
                inf_sign = term_sign
            # Divergent terms have conflicting signs.
            elif inf_sign != term_sign:
                return None
        elif lim_val.is_real:
            continue
        else:
            # Non-real or undefined limits (e.g., zoo) are not supported.
            return None

    # At least one infinite term is required; bounded terms are optional.
    if have_inf:
        return 'const_inf_add'

    return None


def const_inf_mul_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions of the form 'c * f(x)', where:
      - c is a nonzero finite real constant (optional),
      - f(x) tends to +-oo as x approaches the limit point,
      - The expression is a pure product (not a rational expression).

    This matcher excludes cases involving:
      - Oscillatory limits (e.g., AccumBounds),
      - Factors that tend to zero (to avoid indeterminate forms like 0 * oo),
      - Expressions that are internally represented as fractions.
    """
    if not isinstance(expr, Mul):
        return None
    _, den = expr.as_numer_denom()
    # Reject expressions that are rational (i.e., have a nontrivial denominator).
    if den != 1:
        return None
    var, point, direction = get_limit_args(context)
    factors = expr.as_ordered_factors()

    has_inf = False
    for factor in factors:
        lim_val = limit(factor, var, point, dir=direction)
        if lim_val in (oo, -oo):
            has_inf = True
        elif lim_val.is_real:
            # Oscillatory behavior is not allowed.
            if isinstance(lim_val, AccumBounds):
                return None
            # Zero limit would lead to an indeterminate form (0 * oo).
            if lim_val == 0:
                return None
        else:
            # Non-real or undefined limits (e.g., zoo) are not supported.
            return None

    if has_inf:
        return 'const_inf_mul'

    return None


def const_inf_div_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions of the form 'bounded / unbounded', where:
      - The numerator approaches a finite real limit (i.e., is bounded),
      - The denominator diverges to +-oo.

    This pattern corresponds to limits that evaluate to zero due to a bounded
    quantity being divided by an unbounded one.
    """
    var, point, direction = get_limit_args(context)
    num, den = expr.as_numer_denom()
    if den == 1:
        return None

    if is_constant(num, var, point, direction) and is_infinite(den, var, point, direction):
        return 'const_inf_div'
    return None


def const_zero_div_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches expressions of the form 'c / f(x)', where:
      - The numerator tends to a nonzero finite real constant (c != 0),
      - The denominator tends to zero.

    This pattern typically indicates a limit that diverges to +-oo,
    depending on the signs of the numerator and denominator near the limit point.
    """
    var, point, direction = get_limit_args(context)
    num, den = expr.as_numer_denom()
    if den == 1:
        return None

    if is_constant(num, var, point, direction):
        num_lim = limit(num, var, point, dir=direction)
        if num_lim != 0 and is_zero(den, var, point, direction):
            return 'const_zero_div'
    return None


def small_o_add_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    """Matches sums or differences of multiple infinitesimal terms,
    i.e., expressions where every term tends to zero at the limit point.

    This pattern is used to identify o(1) behavior under addition.
    A single infinitesimal term is intentionally excluded—such cases
    should be handled by direct substitution or simpler matchers.
    """
    if not isinstance(expr, Add):
        return None

    var, point, direction = get_limit_args(context)
    terms = expr.as_ordered_terms()
    # Require at least two terms; single-term infinitesimals are handled elsewhere.
    if len(terms) < 2:
        return None
    # All terms must vanish in the limit.
    for term in terms:
        if not is_zero(term, var, point, direction):
            return None  # Non-infinitesimal term found.

    return 'small_o_add'


def conjugate_rationalize_matcher(expr, _context) -> MatcherFunctionReturn:
    """Matches expressions whose numerator is the difference (or sum) of two square roots,
    i.e., of the form (sqrt(A)+-sqrt(B))/D, where A and B are subexpressions.

    This pattern is typically targeted for rationalization via multiplication by the
    conjugate (sqrt(A)+-sqrt(B)).
    """
    num, _ = expr.as_numer_denom()
    # Check if numerator is a sum/difference of exactly two terms.
    if isinstance(num, Add) and len(num.args) == 2:
        a, b = num.args
        # Both terms must be explicit square roots
        if a.func == sqrt and b.func == sqrt:
            return 'conjugate_rationalize'
    return None
