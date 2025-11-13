# TODO f-string 里都是单 $ 包裹

from typing import Tuple

from sympy import (
    Expr, Function, Limit, Mul, Pow, S, Symbol, acos, acot, asin, atan, coth,
    cos, cosh, cot, csc, diff, exp, latex, log, oo, sec, series, simplify,
    sin, sinh, tan, tanh, together
)

from utils import Context, MatcherFunctionReturn, RuleFunctionReturn
from domains.limit import get_limit_args


def _is_zero(expr: Expr) -> bool:
    """Determine whether the given expression is identically zero."""
    return expr == S.Zero


def _is_infinite(expr: Expr) -> bool:
    """Determine whether the given expression is (positive or negative) infinity."""
    return expr in (oo, -oo)


def _extract_num_den(expr: Expr) -> Tuple[Expr, Expr]:
    """Extract the numerator and denominator of an expression in canonical rational form.

    This function first rewrites the input expression using together() to combine
    nested fractions and put it into a single rational form (i.e., A/B where A and B
    are simplified expressions with no embedded divisions). It then uses SymPy's
    .as_numer_denom() to extract the numerator and denominator.

    The result satisfies:
        expr == num / den   (up to simplification)
    and den is guaranteed to be non-zero in symbolic form (though may evaluate to zero
    for specific substitutions).
    """

    combined = together(expr)
    num, den = combined.as_numer_denom()
    return num, den


def _get_type(a: Expr, b: Expr) -> str:
    """Classify the indeterminate form based on the limiting values of numerator and denominator.

    This function identifies common indeterminate forms that arise in limit evaluation,
    specifically:
      - '0/0'       when both numerator and denominator tend to zero,
      - 'oo/oo'     when both tend to (positive or negative) infinity.

    The inputs a and b are expected to be the limit values (not the original expressions),
    typically obtained via _limit_or_series.
    """

    if _is_zero(a) and _is_zero(b):
        return '0/0'
    if _is_infinite(a) and _is_infinite(b):
        return rf"\infty / \infty"
    return "no_match"


def _limit_or_series(expr: Expr, var: Symbol, point: Expr, direction: str) -> Expr | None:
    """Attempt to evaluate the limit of expr as var approaches point from the given direction.

    If direct limit evaluation fails (e.g., due to complexity or unsupported forms),
    fall back to a first-order asymptotic approximation using series expansion.

    The fallback uses the leading term of the series expansion around point.
    If the series contains divergent terms (e.g., oo, -oo, NaN), it returns oo
    as a conservative indicator of divergence.
    """

    # Attempt 1: Direct limit evaluation
    try:
        result = Limit(expr, var, point, dir=direction).doit()
        if result is not S.NaN:
            return result
    except Exception:
        pass  # Proceed to fallback

    # Attempt 2: Series expansion (first non-O term)
    try:
        # Use n=2 to reliably get the leading term; n=1 may return O(...) only
        s = series(expr, var, point, n=2).removeO()
        if s.has(oo, -oo, S.NaN):
            return oo  # Indicate divergence conservatively
        return s
    except Exception:
        return None


def _get_indeterminate_type(numerator: Expr, denominator: Expr, var: Symbol, point: Expr, direction: str) -> str:
    """Determine whether the rational expression numerator/denominator yields an
    indeterminate form of type 0/0 or oo/oo as var approaches point from the given direction.

    To handle limits at infinity, the function applies the standard substitution:
        x to 1/t   as x to oo  (with t to 0+),
        x to 1/t   as x to -oo (with t to 0-).
    This transforms the problem into a finite-limit evaluation at t = 0.

    The function uses _limit_or_series to compute asymptotic values of numerator and denominator,
    then classifies the pair using _get_type.
    """

    # Handle infinite limit points via substitution x = 1/t
    if _is_infinite(point):
        t = Symbol('t', positive=True)  # t > 0 ensures real substitution
        dir_t = '+' if point == oo else '-'
        num_t = numerator.subs(var, 1/t)
        den_t = denominator.subs(var, 1/t)
        a = _limit_or_series(num_t, t, 0, dir_t)
        b = _limit_or_series(den_t, t, 0, dir_t)
    # Finite limit point: evaluate directly
    else:
        a = _limit_or_series(numerator, var, point, direction)
        b = _limit_or_series(denominator, var, point, direction)
    return _get_type(a, b)


def _count_nodes(expr: Expr) -> int:
    """Count the total number of nodes in the expression tree.

    This function recursively traverses the SymPy expression tree and counts
    every node (including leaves and internal operation nodes). The count reflects
    the structural complexity of the expression and can be used as a heuristic
    for simplification cost, pattern matching priority, or result comparison.

    - Atomic expressions (e.g., Symbol, Integer, Pi) have 1 node.
    - For compound expressions, the count is 1 (for the current node) plus the
      sum of counts from all arguments.

    Note:
        This is equivalent to expr.count(lambda x: True) but implemented
        explicitly for clarity and potential performance tuning.

    Args:
        expr (Expr): The SymPy expression to analyze.

    Returns:
        int: Total number of nodes in the expression tree.

    Examples:
        >>> _count_nodes(x)
        1
        >>> _count_nodes(x + y)
        3  # Add(x, y): 1 (Add) + 1 (x) + 1 (y)
        >>> _count_nodes(sin(x))
        2  # sin(x): 1 (sin) + 1 (x)
    """
    if not expr.args:
        return 1
    return 1 + sum(_count_nodes(arg) for arg in expr.args)


def _count_special_functions(expr: Expr) -> int:
    """Count occurrences of predefined elementary special functions in an expression.

    This function recursively traverses the expression tree and counts nodes that are
    instances of common elementary functions, including:
      - Trigonometric: sin, cos, tan, cot, sec, csc
      - Inverse trig: asin, acos, atan, acot
      - Hyperbolic: sinh, cosh, tanh, coth
      - Exponential/logarithmic: exp, log (includes ln)

    Note:
        - log in SymPy represents the natural logarithm (i.e., ln); no separate ln symbol exists.
        - The check uses type identity (isinstance) rather than string matching,
          which is faster, safer, and immune to naming ambiguities or custom subclasses.

    Args:
        expr (Expr): The SymPy expression to analyze.

    Returns:
        int: Total number of special function applications in the expression tree.

    Examples:
        >>> _count_special_functions(sin(x) + exp(x))
        2
        >>> _count_special_functions(x * log(x))
        1
        >>> _count_special_functions(x + y)
        0
    """
    # Define the set of target function classes once (at module level for efficiency)
    SPECIAL_FUNCS = {
        sin, cos, tan, cot, sec, csc,
        asin, acos, atan, acot,
        sinh, cosh, tanh, coth,
        exp, log
    }

    def _count(node: Expr) -> int:
        if isinstance(node, Function):
            # Check if the function class is in our whitelist
            if type(node) in SPECIAL_FUNCS:
                return 1 + sum(_count(arg) for arg in node.args)
        # Recurse into arguments for non-function or non-special-function nodes
        return sum(_count(arg) for arg in node.args) if node.args else 0

    return _count(expr)


def _has_fraction(expr: Expr) -> bool:
    """Check whether the expression contains a fractional (non-polynomial) term.

    A fraction is defined as any subexpression that represents division,
    which in SymPy typically appears as:
      - A power with a negative integer or symbolic exponent (e.g., x**(-1), x**(-n)),
      - Or, more generally, any structure that would render as a denominator.

    This function recursively traverses the expression tree and returns True
    if any such fractional component is found.

    Note:
        - Expressions like 1/x, x**(-2), or sin(x)/x will return True.
        - Pure polynomials (e.g., x+1, x**2) return False.
        - This does not consider rational constants (e.g., Rational(1, 2)) as "fractions"
          in the structural sense-only symbolic denominators.

    Args:
        expr (Expr): The SymPy expression to inspect.

    Returns:
        bool: True if the expression contains a symbolic fraction; False otherwise.
    """

    #  Power with negative exponent yo indicates 1/(something)
    if expr.is_Pow:
        # Check if exponent is negative (works for integers, symbols, and expressions)
        if expr.exp.is_negative is True:
            return True

    # Recurse into arguments
    return any(_has_fraction(arg) for arg in expr.args)


def _count_products(expr: Expr) -> int:
    """Count the number of binary multiplication operations implied by the expression tree.

    SymPy represents multiplication as n-ary Mul nodes (e.g., x*y*z is a single Mul(x, y, z)).
    This function interprets such an n-ary product as (n - 1) binary multiplications,
    which reflects the actual number of multiplication *operations* needed to evaluate it.

    The count is recursive: products inside subexpressions (e.g., in numerators, exponents, etc.)
    are also included.

    Args:
        expr (Expr): The SymPy expression to analyze.

    Returns:
        int: Total number of multiplication operations in the expression tree.

    Examples:
        >>> _count_products(x)
        0
        >>> _count_products(x*y)
        1
        >>> _count_products(x*y*z)
        2  # Mul(x, y, z) to x*y*z requires two multiplications
        >>> _count_products((x*y)+(z*w))
        2  # one in each term
    """
    if expr.is_Mul:
        # An n-ary Mul node implies (n-1) multiplication operations
        return (len(expr.args) - 1) + sum(_count_products(arg) for arg in expr.args)
    # Recurse into all arguments for non-Mul composite expressions (Add, Pow, Function, etc.)
    return sum(_count_products(arg) for arg in expr.args) if expr.args else 0


def _count_powers(expr: Expr) -> int:
    """Count the number of power operations in the expression tree, with increased weight
    for symbolic (variable-containing) powers.

    Each Pow node contributes:
      - 2 points if either base or exponent contains a symbol (indicating non-constant behavior),
      - 1 point if both base and exponent are fully constant (e.g., 2**3, pi**e).

    The score is recursive: nested powers (e.g., (x**y)**z) are counted at every level.

    This metric serves as a heuristic for expression complexity-symbolic powers often
    require more advanced handling in limits, series expansions, or simplifications.

    Args:
        expr (Expr): The SymPy expression to analyze.

    Returns:
        int: Weighted count of power operations.

    Examples:
        >>> _count_powers(2**3)
        1
        >>> _count_powers(x**2)
        2
        >>> _count_powers(x**y)
        2
        >>> _count_powers((x**2)**3)
        3  # outer Pow (symbolic base to +2) + inner Pow (symbolic base to +1? Wait-see note below)
        # Actually: inner = x**2 to 2; outer = (..)**3 to base has x to 2; total = 2 + 2 = 4?
        # But our logic adds recursively: outer returns 2 + count(inner), inner returns 2 to total 4.
        # So example corrected:
        >>> _count_powers((x**2)**3)
        4
    """
    if expr.is_Pow:
        base, _exp = expr.args
        # Determine if the power is "symbolic" (i.e., depends on variables)
        is_symbolic = base.has(Symbol) or _exp.has(Symbol)
        weight = 2 if is_symbolic else 1
        return weight + _count_powers(base) + _count_powers(_exp)

    # Recurse into all arguments for non-Pow composite expressions
    return sum(_count_powers(arg) for arg in expr.args) if expr.args else 0


def _estimate_derivative_complexity(numerator: Expr, denominator: Expr, var: Symbol) -> int:
    """Estimate the structural complexity of the derivative of a quotient.

    This function computes the formal derivative of the ratio numerator/denominator
    using the identity:

        d/d(var) (N/D) = N' / D'   [heuristic approximation for complexity estimation]

    > Note: This is not the true quotient rule derivative ((N'·D - N·D')/D²),
    > but a simplified proxy used solely to gauge relative complexity of applying
    > lhopital's rule or similar transformations. The goal is comparative scoring,
    > not mathematical correctness of the derivative itself.

    The complexity score combines multiple weighted heuristics:
      - Total expression tree size (_count_nodes)
      - Occurrences of special functions (trig, exp, log, etc.)
      - Presence of fractional subexpressions
      - Number of multiplication operations
      - Weighted count of power operations (symbolic powers penalized more)

    Lower scores indicate simpler expressions.

    Args:
        numerator (Expr): Numerator of the original fraction.
        denominator (Expr): Denominator of the original fraction.
        var (Symbol): Differentiation variable.

    Returns:
        int: Non-negative integer representing estimated complexity.

    Example:
        >>> _estimate_derivative_complexity(sin(x), x, x)
        # Computes derivative of sin(x)/x to approx as cos(x)/1 to moderate score
    """
    try:
        # Compute derivative proxies
        num_diff = diff(numerator, var)
        den_diff = diff(denominator, var)

        # Avoid division by zero or undefined forms
        if den_diff == 0:
            # If denominator derivative vanishes, the heuristic breaks down;
            # assign high penalty to discourage this path.
            return 10**6

        # Form the heuristic derivative ratio
        derivative_ratio = num_diff / den_diff

        # Simplify to reduce artificial complexity from unsimplified forms
        # Use cautious simplification to avoid expensive transformations
        result = simplify(derivative_ratio)

    except Exception:
        # Fallback on raw ratio if simplification fails (e.g., due to timeouts or singularities)
        result = num_diff / den_diff

    # Aggregate weighted complexity metrics
    complexity = (
        _count_nodes(result)
        + 2 * _count_special_functions(result)
        + (5 if _has_fraction(result) else 0)
        + _count_products(result)
        + 3 * _count_powers(result)
    )

    return max(0, int(complexity))


def _choose_best_conversion(f: Expr, g: Expr, var: Symbol) -> str:
    """Select the optimal indeterminate form conversion strategy for applying Lhopital's rule.

    Given two expressions f and g that form an indeterminate product like f*g
    (e.g., 0*oo), this function evaluates two canonical rewritings:
      - zero_over_zero: Rewrite as f / (1/g) to 0/0 form
      - inf_over_inf: Rewrite as g / (1/f) to oo/oo form

    The choice is based on a heuristic complexity estimate of the resulting derivative
    ratio after one application of Lhopital's rule. The goal is to minimize symbolic
    explosion and favor simpler subsequent differentiation.

    Strategy:
        1. Prefer the form whose derivative ratio (N'/D') has lower structural complexity.
        2. Avoid conversions that introduce nested fractions, special functions, or high-degree powers.
        3. Favor paths likely to terminate quickly in recursive limit evaluation.

    Args:
        f (Expr): First factor in the indeterminate product (e.g., tends to 0).
        g (Expr): Second factor (e.g., tends to oo).
        var (Symbol): Limit variable.

    Returns:
        str: Either "zero_over_zero" or "inf_over_inf".

    Note:
        This assumes the caller has already confirmed the 0 * oo indeterminate form.
        The function does not validate limits-only compares transformation costs.

    Example:
        For x*log(x) as x to 0+, we have f = x to 0, g = log(x) to -oo.
        Converting to x / (1/log(x)) (0/0) typically yields simpler derivatives
        than log(x) / (1/x) (oo/oo), so "zero_over_zero" is preferred.
    """
    try:
        # Estimate complexity for 0/0 form: f / (1/g) to numerator=f, denominator=1/g
        zero_zero_complexity = _estimate_derivative_complexity(f, 1/g, var)
    except Exception:
        # Fallback to high penalty if transformation fails (e.g., division by zero)
        zero_zero_complexity = float('inf')

    try:
        # Estimate complexity for oo/oo form: g / (1/f) to numerator=g, denominator=1/f
        inf_inf_complexity = _estimate_derivative_complexity(g, 1/f, var)
    except Exception:
        inf_inf_complexity = float('inf')

    # Prefer zero_over_zero in case of tie (often more stable numerically and symbolically)
    if zero_zero_complexity <= inf_inf_complexity:
        return "zero_over_zero"
    else:
        return "inf_over_inf"


def lhopital_direct_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply Lhopital's Rule directly to a limit expression in 0/0 or oo/oo form.

    This rule transforms:
        x to a, f(x)/g(x)  to  x to a f'(x)/g'(x)
    provided the original limit is an indeterminate form of type 0/0 or oo/oo.
    """
    var, point, direction = get_limit_args(context)

    num, den = _extract_num_den(expr)

    typ = _get_indeterminate_type(num, den, var, point, direction)

    try:
        num_deriv = diff(num, var)
        den_deriv = diff(den, var)

        new_expr = Limit(num_deriv/den_deriv, var, point, dir=direction)

        # Map internal type to display form
        display_type = r'\frac{0}{0}' if typ == '0/0' else r'\frac{\infty}{\infty}'

        explanation = (
            f"原式为 ${display_type}$ 型不定式,应用洛必达法则: "
            f"对分子 ${latex(num)}$ 和分母 ${latex(den)}$ 关于 ${latex(var)}$ 分别求导,得到: "
            f"${latex(num_deriv)}, {latex(den_deriv)}$"
            f"因此极限转化为: ${latex(new_expr)}$"
        )

        return new_expr, explanation

    except Exception:
        return None


def lhopital_zero_times_inf_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply Lhopital's Rule to indeterminate forms of type 0 * +-oo.

    This function handles expressions like f * g where one factor tends to 0
    and the other to +-oo as var to point. It:
      1. Identifies which factor is vanishing and which is divergent,
      2. Rewrites the product into a quotient (either 0/0 or oo/oo),
      3. Selects the conversion that minimizes estimated derivative complexity,
      4. Applies Lhopital's Rule once and returns the new limit.
    """
    var, point, direction = get_limit_args(context)

    f, g = expr.args

    # Choose optimal rewriting strategy
    conversion_type = _choose_best_conversion(f, g, var)

    if conversion_type == "zero_over_zero":
        # Rewrite as f / (1/g) to 0/0
        numerator = f
        denominator = 1/g
        display_type = rf'\frac{{0}}{{0}}'

    else:  # conversion_type == "inf_over_inf"
        # Rewrite as g / (1/f) to oo/oo
        numerator = g
        denominator = 1/f
        display_type = rf'\frac{{\infty}}{{\infty}}'

    conversion_explanation = (
        f"原式为 $0 \\cdot \\infty$ 型不定式,转换为 ${display_type}$ 型:"
        f"${latex(expr)} = \\frac{{{latex(numerator)}}}{{{denominator}}}$"
    )

    numerator_diff = diff(numerator, var)
    denominator_diff = diff(denominator, var)

    diff_expr = numerator_diff / denominator_diff
    diff_limit = Limit(diff_expr, var, point, direction)

    explanation = conversion_explanation + (
        f"应用洛必达法则, 分子分母分别求导:"
        f"$\\frac{{d}}{{d{latex(var)}}} \\left({latex(numerator)}\\right) = {latex(numerator_diff)}$"
        f"$\\frac{{d}}{{d{latex(var)}}} \\left({latex(denominator)}\\right) = {latex(denominator_diff)}$"
        f"得到新极限: ${latex(diff_limit)}$"
    )

    return diff_limit, explanation


def lhopital_inf_minus_inf_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Apply Lhopital's Rule to indeterminate forms of type oo-oo.

    This function handles expressions like f-g where both f to +-oo and g to +-oo
    as var to point. It rewrites the difference into a quotient using the identity:
        f-g = (1/g-1/f) / (1/(f*g))

    which yields a 0/0 form under typical conditions (since 1/f to 0 and 1/g to 0).
    The resulting quotient is then processed via one step of Lhopital's Rule.
    """

    var, point, direction = get_limit_args(context)

    f, g = expr.args

    try:
        numerator = 1/g - 1/f
        denominator = 1/(f * g)

        conversion_explanation = (
            f"原式为 $\\infty - \\infty$ 型不定式, 通过代数变形转换为 $\\frac{{0}}{{0}}$ 或 $\\frac{{\\infty}}{{\\infty}}$ 型:"
            f"${latex(expr)} = \\frac{{{latex(numerator)}}}{{{latex(denominator)}}}$"
        )

        numerator_diff = diff(numerator, var)
        denominator_diff = diff(denominator, var)

        diff_expr = numerator_diff / denominator_diff
        diff_limit = Limit(diff_expr, var, point, direction)
        explanation = conversion_explanation + (
            f"应用洛必达法则,分子分母分别求导:"
            f"$\\frac{{d}}{{d{var}}} \\left({latex(numerator)}\\right) = {latex(numerator_diff)}$\\quad"
            f"$\\frac{{d}}{{d{var}}} \\left({latex(denominator)}\\right) = {latex(denominator_diff)}$\\quad"
            f"得到新极限: ${latex(diff_limit)}$"
        )

        return diff_limit, explanation

    except Exception:
        return None


def lhopital_power_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """Handle indeterminate power forms: 0^0, 1^oo, and oo^0.

    This function applies the standard logarithmic transformation:
        f(x)^g(x) = exp(g(x)*ln(f(x)))
    to convert the power expression into an exponential of a product.
    The resulting exponent g(x)*log(f(x)) typically becomes a 0*oo form,
    which can then be handled by other rules (e.g., hopital_zero_times_inf_rule).
    """

    var, point, direction = get_limit_args(context)

    base, exp_arg = expr.args

    lim_base = _limit_or_series(base, var, point, direction)
    lim_exp = _limit_or_series(exp_arg, var, point, direction)

    # Classify indeterminate type
    if _is_zero(lim_base) and _is_zero(lim_exp):
        typ = "0^0"
    elif _is_infinite(lim_base) and _is_zero(lim_exp):
        typ = rf"\infty^0"
    else:
        typ = "1^\\infty"

    # Apply the canonical transformation: f^g = exp(g*log(f))
    transformed_expr = exp_arg * log(base)
    exp_expr = exp(transformed_expr)
    limit_exp_expr = Limit(exp_expr, var, point, direction)

    expr_latex = latex(expr)
    explanation = (
        f"原式为 ${typ}$ 型不定式, 使用指数变换: "
        f"${expr_latex} = e^{{{latex(transformed_expr)}}}$"
        f"因此, $\\lim_{{{latex(var)} \\to {latex(point)}^{{{direction}}}}} {expr_latex} "
        f"= {latex(limit_exp_expr)}$"
    )
    return limit_exp_expr, explanation


def lhopital_direct_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    """Match expressions that are directly in 0/0 or oo/oo indeterminate forms.

    This matcher checks whether the given expression is a non-trivial quotient (i.e., not just a single term)
    and whether its numerator and denominator tend to 0/0 or oo/oo as var to point.
    """
    var, point, direction = get_limit_args(context)

    num, den = _extract_num_den(expr)
    # Reject trivial quotients (e.g., f(x)/1 or just f(x))
    if num == expr and den == 1:
        return None
    typ = _get_indeterminate_type(num, den, var, point, direction)
    return 'lhopital_direct' if typ in ("0/0", rf"\infty/\infty") else None


def lhopital_zero_times_inf_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    """Match expressions of the indeterminate form 0*oo.

    This matcher identifies products f*g where exactly two factors are present,
    and one factor tends to 0 while the other tends to +-oo as var to point.
    """

    if not isinstance(expr, Mul) or len(expr.args) != 2:
        return None

    var, point, direction = get_limit_args(context)

    f, g = expr.args

    lim_f = _limit_or_series(f, var, point, direction)
    lim_g = _limit_or_series(g, var, point, direction)

    if (_is_zero(lim_f) and _is_infinite(lim_g)) or (_is_infinite(lim_f) and _is_zero(lim_g)):
        return 'lhopital_zero_times_inf'

    return None


def lhopital_inf_minus_inf_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    """Match expressions of the indeterminate form oo-oo.

    This matcher identifies binary additive expressions f+g (which may represent f-h
    if g = -h) where both terms diverge to infinity (positive or negative) as
    var to point.
    """

    if not expr.is_Add or len(expr.args) != 2:
        return None

    var, point, direction = get_limit_args(context)

    f, g = expr.args
    lim_f = _limit_or_series(f, var, point, direction)
    lim_g = _limit_or_series(g, var, point, direction)
    return 'lhopital_inf_minus_inf' if _is_infinite(lim_f) and _is_infinite(lim_g) else None


def lhopital_power_matcher(expr: Expr, context: Context) -> MatcherFunctionReturn:
    """Match power expressions of indeterminate forms: 0^0, oo^0, and 1^oo.

    This matcher identifies expressions base^exponent where the asymptotic behavior
    of the base and exponent as var to point yields one of the three classic indeterminate
    power forms. These cases require logarithmic transformation before applying Lhopital's Rule.
    """

    if not isinstance(expr, Pow):
        return None

    var, point, direction = get_limit_args(context)

    base, _exp = expr.args
    lim_base = _limit_or_series(base, var, point, direction)
    lim_exp = _limit_or_series(_exp, var, point, direction)
    if lim_base is None or lim_exp is None:
        return None

    base_zero = _is_zero(lim_base)
    exp_zero = _is_zero(lim_exp)
    base_inf = _is_infinite(lim_base)
    exp_inf = _is_infinite(lim_exp)
    base_is_one = lim_base == S.One

    # Match the three canonical indeterminate power forms
    if (base_zero and exp_zero) or \
       (base_inf and exp_zero) or \
       (base_is_one and exp_inf):
        return "lhopital_power"
    return None
