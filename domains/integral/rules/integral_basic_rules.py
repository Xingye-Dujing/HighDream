from sympy import (
    Add, Expr, Eq, Integral, Mul,  Pow, atan, degree, div, fraction,
    integrate, powsimp, latex, radsimp, solve, simplify, symbols, sqrt
)

from utils import MatcherFunctionReturn, RuleContext, RuleFunctionReturn, can_use_weierstrass
from utils.latex_formatter import wrap_latex


def handle_poly(num: Expr, den: Expr, var: Expr) -> tuple[Expr, str, bool]:
    """Handle polynomial expressions for integration"""
    # If expr is rational fraction, performing partial fraction decomposition.
    expr_copy = num / den
    expr_apart = expr_copy.apart()

    # If the rational fraction is reducible
    if expr_apart != expr_copy:
        return expr_apart, "(有理分式)化为真分式或部分分式分解", False

    # If the rational fraction is not reducible:
    # 1. irreducible linear poly:
    if degree(den) == 1:
        q, r = div(num, den)
        result = q + r/den
        return result, "裂项(分母为一次多项式)", False

    # 2. irreducible quadratic poly:
    if degree(den) == 2:
        # The simplest irreducible quadratic poly, no decomposition required
        if num.is_constant():
            result = integrate(expr_copy, var)
            if result.has(atan):
                return result, \
                    rf"(分母为二次多项式且分子为常数)分母配方/提公因子凑 $ \frac{{1}}{{b}} \frac{{1}}{{u^2+1}} $ 法", True
            # If square roots are needed during decomposition, another algorithm should be used
            return expr_copy.apart(full=True).doit(), "(有理分式)化为真分式或部分分式分解", False

        den_diff = den.diff()
        alpha, beta = symbols('alpha beta')
        # Construct the identity: num = alpha * den_diff + beta
        sol = solve(Eq(num, alpha * den_diff + beta),
                    (alpha, beta))
        alpha_val, beta_val = sol[alpha], sol[beta]

        part1 = alpha_val*den_diff/den
        part2 = beta_val/den
        expr_copy = part1 + part2
        return expr_copy, rf"(分母为不可约二次) $构造等式(分子 = \alpha \cdot 分母导数 + \beta)进行裂项$", False

    # Cases like 1/(x**2+1)**2 that have already been reduced to the simplest form will reach here
    return None, None, False


def add_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the sum rule：(f+g) dx = f dx + g dx"""
    var = context['variable']
    var_latex, expr_latex = wrap_latex(var, expr)
    prefix = "应用加法展开规则"
    expr_copy = expr.expand()

    if isinstance(expr, (Mul, Pow)):
        num, den = fraction(expr)
        # If expr is rational fraction, make sure it is a true fraction.
        q, r = div(num, den)
        result = q + r/den

        used = False
        if den != 1 and num.is_polynomial() and den.is_polynomial():
            # The simplest linear poly, no decomposition required
            if degree(den) == 1 and not isinstance(result, Add):
                return None
            if expr == 1/(var**2+1):
                return None
            # If expr is rational fraction, performing partial fraction decomposition.
            result, prefix, is_direct_return = handle_poly(num, den, var)
            # 1. Have already been reduced to the simplest form
            if not result:
                return None
            # 2. The simplest irreducible quadratic poly, directly return result
            if is_direct_return:
                return result, prefix
            # 3. If there is only one term after decomposition
            if not isinstance(result, Add):
                new_expr = Integral(result, var)
                return new_expr, f"{prefix}: $\\int {expr_latex}\\,d {var_latex} = {latex(new_expr)}$"
            # 4. The rational fraction is reducible
            used = True
            expr_copy = result

        if not used and den != 1:
            den_diff = den.diff()
            alpha, beta = symbols('alpha beta')
            try:
                # Construct the identity: num = alpha * den_diff + beta
                sol = solve(Eq(num, alpha * den_diff + beta*den),
                            (alpha, beta))
                alpha_val, beta_val = sol[alpha], sol[beta]
                if alpha_val.is_constant() and beta_val.is_constant():
                    part1 = radsimp(alpha_val*den_diff/den)
                    part2 = beta_val
                    result = Integral(part1, var) + Integral(part2, var)
                    return result, f"$构造等式(分子 = \\alpha \\cdot 分母导数 + \\beta \\cdot 分母)进行裂项: \\int {expr_latex}\\,d {var_latex} = {latex(result)}$"
            except Exception as e:
                print(f"Error in add_rule: {e}")

        # Handle the expressions containing non-polynomial elements such as log
        if not used:
            result = result.expand()
            if isinstance(result, Add):
                expr_copy = result
                used = True

        if not used:
            return None

    # Comparing to diff's add_rule, we use list comprehension.
    integrals = []
    for term in expr_copy.args:
        if can_use_weierstrass(term, var):
            integrals.append(Integral(radsimp(powsimp(term)), var))
        else:
            integrals.append(Integral(simplify(powsimp(term)), var))
    new_expr = Add(*integrals)

    explanation = f"{prefix}: $\\int {expr_latex}\\,d {var_latex} = {latex(new_expr)}$"
    return new_expr, explanation


def mul_const_rule(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    """Apply the const mul rule: c*f(x) dx = c * f(x) dx"""
    used = False

    var = context['variable']
    coeff_first, expr_copy = expr.as_coeff_Mul(rational=True)
    num, den = fraction(expr_copy)
    if den == 1:
        coeff, func_part = expr_copy.as_content_primitive(radical=True)
    else:
        coeff_num, func_num = num.as_content_primitive(radical=True)
        coeff_den, func_den = den.as_content_primitive(radical=True)
        if coeff_num != 1 or coeff_den != 1:
            used = True
        coeff = simplify(coeff_num/coeff_den)
        func_part = simplify(func_num/func_den)

    coeff *= coeff_first

    if isinstance(func_part, Mul) and func_part.args[0].is_constant():
        coeff *= func_part.args[0]
        func_part = func_part/coeff

    if not expr.has(sqrt) and coeff.has(sqrt):
        coeff, func_part = expr.as_coeff_Mul()

    var_latex, expr_latex, func_part_latex = wrap_latex(var, expr, func_part)
    if coeff == -1:
        return -Integral(func_part, var), f"负号提出: $\\int {expr_latex}\\,d{var} = -\\int {func_part_latex}\\,d{var_latex}$"
    if coeff != 1:
        return coeff * Integral(func_part, var), f"常数因子提取: $\\int {expr_latex}\\,d{var} = {latex(coeff)} \\int {func_part_latex}\\,d{var_latex}$"
    if used:
        return Integral(func_part, var), f"约分: $\\int {expr_latex}\\,d{var} = \\int {func_part_latex}\\,d{var_latex}$"

    return None


def add_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    if isinstance(expr, (Add, Mul)):
        return 'add'
    # 1/(poly^n): poly^(-n)
    if isinstance(expr, Pow) and expr.exp.is_integer and expr.base != context['variable']:
        return 'add'
    return None


def mul_const_matcher(expr: Expr, context: RuleContext) -> MatcherFunctionReturn:
    if not can_use_weierstrass(expr, context['variable']):
        return 'mul_const'
