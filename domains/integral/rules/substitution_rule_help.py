from sympy import (
    Add, Abs, Dummy, Eq, Expr, Function, I, Integral, Pow, Rational, Symbol, Wild,
    acos, asin, atan, cancel, cos, cosh, coth, cot, csc, csch, diff, expand_trig, exp,
    fraction, latex, log, preorder_traversal, powdenest, sech, sec, simplify, sin, sinh,
    solve, sqrt, symbols, tan, tanh, together
)
import math

from core import BaseStepGenerator
from utils import RuleFunctionReturn

sqrt_pow = Rational(1, 2)


def _simp(node: Expr) -> Expr:
    """Cheap canonicalization: ``cancel`` for a pure rational function, else ``simplify``.

    ``simplify`` runs transcendental/trig/exp strategies that are wasted on the many
    rational ratios in the substitution loop (and dominate their cost — e.g. the degree-6
    partial-fraction of 1/(x^4+1)). ``cancel`` gives the same reduced-rational result for
    rational expressions far faster, while the constant/zero tests below (``== 0``,
    ``.has(var)``, ``.free_symbols``) behave identically on a cancelled form. Falls back to
    ``simplify`` for anything with non-rational content, so results are unchanged there."""
    try:
        if node.is_rational_function():
            return cancel(node)
    except (TypeError, AttributeError, NotImplementedError):
        pass
    return simplify(node)


def try_standard_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Attempt standard u-substitution for integrals of the form f(g(x)) g'(x) dx.

    Matches expressions where:
      - A unary function f(g(x)) is present (e.g., sin(x^2), log(cos(x)), exp(tan(x))).
      - The derivative g'(x) appears as a factor (up to a constant multiple)

    Returns the integrated result and explanation if successful.
    """
    # Normalize expression into a list of factors (even for non-Mul)
    expr = together(expr)
    factors = list(expr.args) if expr.is_Mul else [expr]

    for factor_ in factors:
        # Structural "is constant / is the bare variable" test: .is_constant()/.equals()
        # fall back to numeric evalf, which recurses infinitely on some exprs
        # (e.g. asin(x)/sqrt(1-x^2)). free_symbols / == answer the same question safely.
        if not factor_.args or not factor_.free_symbols or factor_ == var:
            continue

        # Extract internal factors for subsequent traversal
        flag = False
        # Look for unary functions like 3^g(x), etc.
        if factor_.is_Pow and factor_.args[1].has(var):
            flag = True
        # Look for unary functions like sin(g(x)), log(g(x)),1/g(x)^2 etc.
        inner = factor_.args[1] if flag else factor_.args[0]

        term_list = [factor_]
        # If it's 1/den, extract den specially, e.g. x/sqrt(x^2+1) extracts sqrt(x^2+1)
        num, den = fraction(factor_)
        if num.equals(1):
            term_list.append(den)
        term_list += list(preorder_traversal(inner))

        for original_term in term_list:
            # Preventing infinite loops (structural: avoids evalf recursion).
            if (not isinstance(original_term, Expr) or original_term == var
                    or original_term == -var or not original_term.free_symbols):
                continue

            check_list = [original_term]
            # Introducing sqrt_term is to handle implicit f(x)^2 cases like x/(x**4+1), x**x*(log(x)+1)/(x**(2*x)+1)
            try:
                sqrt_term = sqrt(original_term)
                # Use a temporary variable with positive real assumptions to aid radical simplification.
                _t = Dummy('t', real=True, positive=True)
                sqrt_term = simplify(sqrt_term.subs(var, _t)).subs(_t, var).replace(
                    Abs, lambda arg: arg)
            except Exception:  # pylint: disable=broad-exception-caught
                sqrt_term = None
            if (sqrt_term is not None and not sqrt_term.has(I)
                    and sqrt_term != var and sqrt_term != -var):
                check_list.append(sqrt_term)

            for term in check_list:

                gp = _simp(diff(term, var))  # g'(x)
                if gp == 0:  # g'(x) = 0
                    continue

                initial_ratio = _simp(expr / gp)
                # Special case: g'(x) dx, let u = g(x), g'(x) dx = 1 du
                if not initial_ratio.free_symbols:
                    # Substitute u = g(x)
                    u = step_gene.get_available_sym(var)
                    step_gene.subs_dict[u] = term
                    explanation = (
                        f"换元法: 令 ${u.name} = {latex(term)}$, $d{u.name} = {latex(gp)}\\,d{var.name}$, "
                        f"原式$ \\int {latex(expr)}\\,d{var.name}$ 化为 $ \\int {latex(initial_ratio)}\\,d{u.name}$"
                    )
                    return Integral(initial_ratio, u), explanation

                try:
                    # Compute the “remaining part” = expr / factor
                    outer_part = expr / factor_  # The rest of the integrand

                    # Check if outer_part = k * g'(x) for some k independent of the
                    # integration variable. On single-variable input free_symbols and
                    # has(var) agree; with a parameter present only has(var) correctly
                    # treats e.g. -1/a (k for exp(-a*x)) as a valid constant multiple.
                    ratio = _simp(outer_part / gp)
                    if ratio.has(var):
                        continue

                    # Substitute u = g(x)
                    u = step_gene.get_available_sym(var)

                    # Construct f(u)
                    f_u = _simp((initial_ratio / ratio).subs(term, u))

                    if f_u.has(var):
                        continue

                    step_gene.subs_dict[u] = term
                    new_expr = ratio * Integral(f_u, u)

                    ratio_latex = '' if ratio.equals(1) else (
                        '-' if ratio.equals(-1) else latex(ratio))
                    gp_latex = '' if gp.equals(1) else (
                        '-' if gp.equals(-1) else latex(gp))

                    explanation = (
                        f"换元法: 令 ${u.name} = {latex(term)}$, $d{u.name} = {gp_latex}\\,d{var.name}$, "
                        f"原式$ \\int {latex(expr)}\\,d{var.name}$ 化为 $ {ratio_latex} \\int {latex(f_u)}\\,d{u.name}$"
                    )

                    return new_expr, explanation

                except (ZeroDivisionError, ValueError, TypeError, NotImplementedError):
                    # Safely skip problematic cases (e.g., division by zero, unsupported ops)
                    # print(f"Warning: Standard substitution failed for {e}")
                    continue

    return None


def _collapse_multiple_angle(e: Expr, theta: Symbol) -> Expr:
    """If ``e`` contains a multiple-angle trig term (sin/cos/... of k*theta, |k|>1),
    expand it back to powers of sin(theta)/cos(theta). The rule library matches
    single-angle powers; a collapsed form like ``8*cos(t)/(1-cos(4*t))`` (which a
    trig substitution can produce) otherwise sends the BFS into a runaway search.
    Returns ``e`` unchanged when there is no multiple angle, so normal substitutions
    are untouched."""
    for atom in e.atoms(Function):
        if atom.func in (sin, cos, tan, cot, sec, csc) and atom.args:
            coeff = atom.args[0].as_coefficient(theta)
            if coeff is not None and coeff not in (-1, 1):
                try:
                    return cancel(expand_trig(e))
                except Exception:  # noqa
                    return e
    return e


def _flip_neg_trig_powers(e: Expr, theta: Symbol) -> Expr:
    """Rewrite negative integer trig powers so the rule library's positive-power matchers
    fire: cos(theta)^(-k) -> sec(theta)^k, sin^-k -> csc^k, tan^-k -> cot^k (k positive int).

    Trig substitution of e.g. (1 - x^2)^(-5/2) yields cos(theta)^(-4); sec_power_matcher
    expects base sec and cos_power_matcher expects a positive exponent, so the raw form
    fell through to Weierstrass (~8s). Flipping to sec(theta)^4 lets the fast reduction
    formula handle it. No effect when there are no negative trig powers (existing +/-1/2,
    3/2 cases are untouched)."""
    def _query(node):
        return (isinstance(node, Pow) and node.exp.is_Integer and node.exp < 0
                and node.base in (cos(theta), sin(theta), tan(theta)))

    def _value(node):
        k = int(-node.exp)
        if node.base == cos(theta):
            return sec(theta) ** k
        if node.base == sin(theta):
            return csc(theta) ** k
        return cot(theta) ** k  # node.base == tan(theta), guaranteed by _query
    return e.replace(_query, _value)


def try_exp_product_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Substitution u = x·eˣ for the classic 考研 trick where the integrand is
    (x+1)·eˣ · R(x·eˣ) up to the missing eˣ factor, e.g. (x+1)/(x(1+x·eˣ)).

    Deliberately narrow: it only ever tries the single inner function g = var·e^var,
    so it cannot fire on (or regress) expressions without e^x.
    """
    if not expr.has(exp):
        return None
    g = var * exp(var)
    gp = diff(g, var)  # (var + 1)·e^var
    ratio = simplify(expr / gp)
    if ratio.has(Integral) or not ratio.has(g):
        return None
    u = step_gene.get_available_sym(var)
    f_u = simplify(ratio.subs(g, u))
    if (f_u != ratio and not f_u.has(var) and not f_u.has(Integral)
            and f_u.is_rational_function(u)):
        step_gene.subs_dict[u] = g
        explanation = (
            f"换元法(凑 x·eˣ): 令 ${u.name} = {latex(g)}$, $d{u.name} = {latex(gp)}\\,d{var.name}$, "
            f"原式$ \\int {latex(expr)}\\,d{var.name}$ 化为 $ \\int {latex(f_u)}\\,d{u.name}$"
        )
        return Integral(f_u, u), explanation
    return None


def try_trig_odd_power(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Odd-power trig product substitution: ``sin^m(x)·cos^n(x)`` with m,n >= 1 integers
    and at least one of m,n odd → a polynomial via ``u=sin x`` (n odd) or ``u=cos x`` (m odd).

    This is the standard textbook technique that Weierstrass (t=tan(x/2)) should never
    preempt: sin^2·cos^5 via Weierstrass explodes to a degree-8 rational (25 steps, ~9s),
    but u=sin gives the polynomial ``u^2(1-u^2)^2`` integrated in milliseconds.

    Narrow by design: fires only on a pure ``Mul`` of ``sin^m`` and ``cos^n`` (both present,
    integer exponents >= 1, at least one odd) with no other factor, so it cannot touch a
    bare ``sin^n``/``cos^n`` (handled by their own matchers) or any mixed expression.
    """
    if not expr.is_Mul:
        return None
    sin_pow = cos_pow = 0
    for factor in expr.args:
        base, expo = factor.as_base_exp()
        if not (expo.is_Integer and expo > 0):
            return None
        if base == sin(var):
            sin_pow += int(expo)
        elif base == cos(var):
            cos_pow += int(expo)
        else:
            return None  # an extraneous factor -> not a pure trig power product
    if sin_pow < 1 or cos_pow < 1:
        return None  # needs both sin and cos; single powers belong to other matchers
    u = step_gene.get_available_sym(var)
    if cos_pow % 2 == 1:
        # keep one cos·dx = du, cos^(n-1) = (1 - sin^2)^((n-1)/2)
        poly = (u ** sin_pow * (1 - u ** 2) ** ((cos_pow - 1) // 2)).expand()
        step_gene.subs_dict[u] = sin(var)
        return Integral(poly, u), (
            f"三角奇次幂换元: 余弦幂为奇, 令 ${u.name} = \\sin({latex(var)})$, $d{u.name} = "
            f"\\cos({latex(var)})\\,d{var.name}$, $\\cos^2 = 1-\\sin^2$ 展开为多项式 $\\int {latex(poly)}\\,d{u.name}$")
    if sin_pow % 2 == 1:
        # keep one sin·dx = -du, sin^(m-1) = (1 - cos^2)^((m-1)/2)
        poly = (-(1 - u ** 2) ** ((sin_pow - 1) // 2) * u ** cos_pow).expand()
        step_gene.subs_dict[u] = cos(var)
        return Integral(poly, u), (
            f"三角奇次幂换元: 正弦幂为奇, 令 ${u.name} = \\cos({latex(var)})$, $d{u.name} = "
            f"-\\sin({latex(var)})\\,d{var.name}$, $\\sin^2 = 1-\\cos^2$ 展开为多项式 $\\int {latex(poly)}\\,d{u.name}$")
    return None  # both even -> double-angle reduction, not this rule


_HYPERBOLIC_FUNCS = (sinh, cosh, tanh, coth, sech, csch)


def try_hyperbolic_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Exponential substitution t = e^var for integrands that are a rational function of
    the hyperbolic functions (sinh/cosh/tanh/...).

    Every hyperbolic function is a rational function of e^x (cosh = (e^x + e^-x)/2, ...),
    so such integrands reduce to a rational function in t and integrate by partial
    fractions. This is the textbook method; without it these fall through to a poor
    integration-by-parts that cycles back to the original integral (e.g. 1/(cosh x + 1)),
    wasting ~10s and producing a degenerate, non-progressing derivation.

    Narrow by design: fires only when a hyperbolic function is present and the
    exp-rewritten, combined expression is genuinely a rational function of e^var, so it
    cannot hijack trig (Weierstrass) or already-rational-in-exp cases handled elsewhere.
    """
    if not expr.has(*_HYPERBOLIC_FUNCS):
        return None
    rw = together(cancel(expr.rewrite(exp)))
    if not rw.is_rational_function(exp(var)):
        return None
    t = step_gene.get_available_sym(var)
    # Replace every e^(k*var) (integer k, including negative) by t**k.
    reps = {}
    for e in rw.atoms(exp):
        coef = e.args[0].as_coefficient(var)
        if coef is not None and coef.is_Integer:
            reps[e] = t ** int(coef)
    if not reps:
        return None
    f_t = simplify(rw.subs(reps) / t)  # dx = dt/t  (dt = e^x dx = t dx)
    if f_t.has(var) or not f_t.is_rational_function(t):
        return None
    step_gene.subs_dict[t] = exp(var)
    return Integral(f_t, t), (
        f"双曲函数化为指数: 令 ${t.name} = e^{{{var.name}}}$, $d{var.name} = d{t.name}/{t.name}$, "
        f"原式$\\int {latex(expr)}\\,d{var.name}$ 化为 $\\int {latex(f_t)}\\,d{t.name}$")


def try_tangent_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Tangent substitution t = tan(var) for a *proper rational* integrand that is even in
    both sin and cos — i.e. a rational function R(sin^2, cos^2) (so f(-sin,-cos)=f).

    Then sin^2 = t^2/(1+t^2), cos^2 = 1/(1+t^2), sin*cos = t/(1+t^2), dx = dt/(1+t^2), which
    gives a plain rational function in t. This is cheaper than Weierstrass (t = tan(x/2)):
    e.g. 1/(sin^2 x + 3) -> ∫dt/(4 t^2 + 3) (degree 2, instant atan) instead of Weierstrass's
    degree-6 rational (~5.6s).

    Guarded to avoid regressing fast paths: fires only when (a) there is a genuine
    denominator involving sin/cos (so bare sin^n/cos^n powers keep their existing
    double-angle / reduction handling), and (b) the integrand is invariant under
    sin -> -sin, cos -> -cos. Odd combos like 1/(sin + cos) correctly fail the parity test.
    """
    if not (expr.has(sin(var)) or expr.has(cos(var))):
        return None
    den = fraction(together(expr))[1]
    if den == 1 or not den.has(sin(var), cos(var)):
        return None  # polynomial in sin/cos (or no trig denominator) -> other matchers
    negated = expr.subs({sin(var): -sin(var), cos(var): -cos(var)})
    if _simp(expr - negated) != 0:
        return None  # not even in sin & cos -> not a function of sin^2/cos^2 alone
    t = step_gene.get_available_sym(var)
    s2 = t ** 2 / (1 + t ** 2)
    c2 = 1 / (1 + t ** 2)
    f_t = expr.subs({sin(var) ** 2: s2, cos(var) ** 2: c2,
                     sin(var) * cos(var): t / (1 + t ** 2),
                     tan(var): t, cot(var): 1 / t})
    f_t = _simp(f_t / (1 + t ** 2))  # * dx/dt
    if f_t.has(var) or not f_t.is_rational_function(t):
        return None
    step_gene.subs_dict[t] = tan(var)
    return Integral(f_t, t), (
        f"正切代换(被积式为 sin²/cos² 的有理式): 令 ${t.name} = \\tan({var.name})$, "
        f"$\\sin^2 = {latex(s2)}$, $\\cos^2 = {latex(c2)}$, $d{var.name} = d{t.name}/(1+{t.name}^2)$, "
        f"化为 $\\int {latex(f_t)}\\,d{t.name}$")


def try_tan_sec_power_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """tan^m(x)·sec^n(x) power substitution (cos(x)^(-k) is read as sec(x)^k).

    Classic cases, both reducing to a polynomial:
      - m odd  -> u = sec x (save sec·tan; tan^(m-1) = (sec^2 - 1)^((m-1)/2))
      - n even -> u = tan x (save sec^2; sec^(n-2) = (1 + tan^2)^((n-2)/2))
    Otherwise (m odd & n even both false) there is no single-step reduction -> return None.

    Without this, ∫tan^3(x)/cos(x) dx (= ∫tan^3·sec) is grabbed by Weierstrass and blown up
    to a degree-8 partial fraction (55 steps, ~8s) instead of one u=sec line.
    """
    factors = list(expr.args) if expr.is_Mul else [expr]
    tan_e = sec_e = 0
    for f in factors:
        base, expo = f.as_base_exp()
        if not expo.is_Integer:
            return None
        e = int(expo)
        if base == tan(var):
            tan_e += e
        elif base == sec(var):
            sec_e += e
        elif base == cos(var):
            if e < 0:
                sec_e += -e  # cos^(-k) == sec^k
            else:
                return None  # positive cos power -> not this pattern
        else:
            return None
    if tan_e < 1 or sec_e < 1:
        return None  # needs both; pure powers belong to tan_power / sec_power matchers
    t = step_gene.get_available_sym(var)
    if tan_e % 2 == 1:
        # u = sec x, du = sec x tan x dx
        rest = ((t ** 2 - 1) ** ((tan_e - 1) // 2) * t ** (sec_e - 1)).expand()
        step_gene.subs_dict[t] = sec(var)
        return Integral(rest, t), (
            f"正切正割幂换元(正切幂为奇): 令 ${t.name} = \\sec({var.name})$, $d{t.name} = "
            f"\\sec\\tan\\,d{var.name}$, $\\tan^2 = {t.name}^2 - 1$ 化为 $\\int {latex(rest)}\\,d{t.name}$")
    if sec_e % 2 == 0:
        # u = tan x, du = sec^2 x dx
        rest = (t ** tan_e * (1 + t ** 2) ** ((sec_e - 2) // 2)).expand()
        step_gene.subs_dict[t] = tan(var)
        return Integral(rest, t), (
            f"正切正割幂换元(正割幂为偶): 令 ${t.name} = \\tan({var.name})$, $d{t.name} = "
            f"\\sec^2\\,d{var.name}$, $\\sec^2 = 1+{t.name}^2$ 化为 $\\int {latex(rest)}\\,d{t.name}$")
    return None


def try_trig_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """
    Apply trigonometric substitution for integrals containing a half-integer power of a
    quadratic radical:
      - (a^2 − x^2)^(k/2)  ->  x = a*sin(theta)
      - (a^2 + x^2)^(k/2)  ->  x = a*tan(theta)
      - (x^2 − a^2)^(k/2)  ->  x = a*sec(theta)
    k any odd integer (±1, ±3, ±5, ...), i.e. exponent ±1/2, ±3/2, ±5/2, ... .

    Uses pattern matching with Wild symbols to extract constant 'a' and exponent 'p'.

    Returns the integrated result and explanation if successful."""

    # The patterns below are (a ± var^2)^(k/2) — a radical whose exponent is a half-integer
    # (denominator 2): ±1/2, ±3/2, ±5/2, ... . cancel() never creates such an exponent from
    # non-radical input, so if no Pow atom has a denominator-2 exponent none can match.
    # Short-circuit to skip the costly cancel() + Wild .find scans on the many compound
    # (log/atan/polynomial) nodes that reach substitution for every BFS step. (has(sqrt) is
    # useless — SymPy stores sqrt(..) as Pow(.., Rational(1,2)), not a sqrt instance.)
    if not any(p.exp.is_Rational and p.exp.q == 2 for p in expr.atoms(Pow)):
        return None

    # If a quadratic radical is trapped *inside a logarithm* (e.g. log(x + sqrt(x^2+1))
    # = asinh(x)), trig substitution cannot rationalize the log argument — it only makes
    # the integrand far harder (x = tan(u) turns it into log(tan u + sec u)/cos^2 u, a 70s
    # detour). Such integrands are handled cleanly by integration by parts, so skip the
    # trig substitution and let parts_rule take over.
    if any(any(p.exp.is_Rational and p.exp.q == 2 for p in L.atoms(Pow))
           for L in expr.atoms(log)):
        return None

    # Note: cancel() is necessary:
    # Simplify to the lowest terms to enhance matching capability.
    # (e.g., sqrt(x**2-8)/(x**2-8) to 1/sqrt(x**2-8))
    expr = cancel(expr)

    # Assume a > 0 and theta is real to help SymPy's simplify:
    # These assumptions are necessary for Integral to work correctly !!!
    # Textbook convention: the constant in a^2 ± x^2 is assumed positive. Accept a
    # bound value unless it is *provably* negative/zero, so a parameter like ``a**2``
    # (whose .is_positive is None) still routes here instead of falling through to the
    # radical substitution (which picks a sign-inconsistent branch for sqrt(a^2 - x^2)
    # and ships a wrong antiderivative). Single-variable (numeric-constant) inputs are
    # unaffected: their .is_positive is already True/False.
    a = Wild('a', exclude=[var], properties=[lambda x: x.is_positive is not False])
    # Half-integer radical exponent (±1/2, ±3/2, ±5/2, ...); the substitution below is
    # algebraically identical for any such power (the quadratic becomes cos^k/sec^k etc.).
    p = Wild('p', exclude=[var], properties=[lambda z: z.is_Rational and z.q == 2])
    # Theta is real and positive to help simplify
    theta = step_gene.get_available_sym(var, real=True, positive=True)

    # Helper: attempt substitution given pattern, sub_expr, d(x)/d(theta), and back-substitution
    def _apply_trig_sub(pattern, x_of_theta, theta_of_x):
        matches = expr.find(pattern)
        if not matches:
            return None

        # Take the first match (enough for teaching context)
        matched_sqrt = list(matches)[0]
        # Extract 'a' by solving matched_sqrt == pattern
        sol = matched_sqrt.match(pattern)
        if not sol or a not in sol:
            return None
        a_val = sol[a]

        x_sub = x_of_theta.subs(a, a_val)
        dx_dtheta = diff(x_sub, theta)

        try:
            # Note: inverse=True is necessary to handle cases like asin(sin(x)) == x
            new_expr = simplify(expr.subs(var, x_sub) * dx_dtheta, inverse=True).replace(
                Abs, lambda arg: arg)
            # Collapse any multiple-angle form (e.g. 8*cos(t)/(1-cos(4t))) back to
            # single-angle powers so the rule library can integrate it quickly.
            new_expr = _collapse_multiple_angle(new_expr, theta)
            # Turn cos/sin/tan negative powers into sec/csc/cot positive powers so the
            # reduction-formula matchers fire instead of falling through to Weierstrass.
            new_expr = _flip_neg_trig_powers(new_expr, theta)
            res = Integral(new_expr, theta)
            step_gene.subs_dict[theta] = theta_of_x.subs(a, sqrt(a_val))

            explanation = (
                f"三角代换：被积式含 ${latex(matched_sqrt)}$, 令 ${latex(var)} = {latex(x_sub)}$,"
                f"则 $d{latex(var)} = \\left({latex(dx_dtheta)}\\right)\\,d{latex(theta)}$, 积分化为 "
                f"${latex(res)}$"
            )
            return res, explanation

        except Exception as e:
            print(f"Error in try_trig_substitution: {e}")
            return None

    # Case 1: (a^2 − x^2)^(k/2)  ->  x = a*sin(theta)
    pattern1 = (a - var ** 2) ** p
    sub1 = sqrt(a) * sin(theta)
    back1 = asin(var / a)
    result = _apply_trig_sub(pattern1, sub1, back1)
    if result:
        return result

    # Case 2: (a^2 + x^2)^(k/2)  ->  x = a*tan(theta)
    pattern2 = (a + var ** 2) ** p
    sub2 = sqrt(a) * tan(theta)
    back2 = atan(var / a)
    result = _apply_trig_sub(pattern2, sub2, back2)
    if result:
        return result

    # Case 3: (x^2 − a^2)^(k/2)  ->  x = a*sec(theta)
    pattern3 = (var ** 2 - a) ** p
    sub3 = sqrt(a) * sec(theta)
    # Alternative: asec(var/a), but acos(a/x) is more common in textbooks.
    back3 = acos(a / var)
    result = _apply_trig_sub(pattern3, sub3, back3)
    if result:
        return result

    return None


def try_undetermined_coeffs_for_radicals(expr: Expr, var: Symbol) -> RuleFunctionReturn:
    """Attempt to solve integrals of the form (P(x))/(sqrt(ax^2+bx+c)) dx using undetermined coefficients' method.

    For integrals of the form (P(x))/(sqrt(ax^2+bx+c)) dx where P(x) is a polynomial,
    we assume the antiderivative has the form Q(x)*sqrt(ax^2+bx+c) + K ((1/sqrt(ax^2+bx+c)) dx)
    where Q(x) is a polynomial of degree deg(P)-1, and K is a constant."""

    # Check if expression is of the form P(x)/sqrt(ax^2+bx+c)
    num, den = fraction(expr)
    # Cheap structural tests first: the denominator must be sqrt(...) i.e. a Pow with
    # exponent 1/2. Only then test the (potentially large) numerator with equals().
    # Same logical condition as before, just reordered so the costly num.equals(1) is
    # skipped on the many non-radical nodes this rule is consulted for.
    if not isinstance(den, Pow) or list(den.args)[1] != sqrt_pow:
        return None
    if num.equals(1):
        return None

    # Extract ax^2 + bx + c from sqrt(ax^2+bx+c)
    radicand = list(den.args)[0]

    # Check if it's a quadratic expression ax^2 + bx + c
    if not radicand.is_Add or len(radicand.args) < 2 or len(radicand.args) > 3:
        return None
    quad_poly = radicand.expand()
    if not quad_poly.is_polynomial(var) or quad_poly.as_poly(var).degree() != 2:
        return None

    # Check if numerator is a polynomial (any degree)
    if not num.is_polynomial(var):
        return None
    num_degree = num.as_poly(var).degree()
    if num_degree < 0:  # Constant numerator (degree 0)
        return None

    # Generate polynomial Q(x) of degree (num_degree - 1)
    # Q(x) = C0 + C1*x + C2*x**2 + ... + C_{m}*x**m where m = num_degree - 1
    m = num_degree - 1
    coeffs = symbols(f'C0:{m + 1}')  # C0, C1, ..., Cm
    Q = sum(coeffs[i] * var ** i for i in range(m + 1))
    if isinstance(Q, int):
        return None

    # Constant K for the integral term
    K = Symbol('K')

    # Assumed antiderivative: Q(x)*sqrt(radicand) + K * ∫(1/sqrt(radicand)) dx
    assumed_antiderivative = Q * sqrt(radicand) + K * Integral(1 / sqrt(radicand), var)

    # Differentiate assumed antiderivative
    diff_assumed = simplify(diff(assumed_antiderivative, var).expand())

    # Clear denominator by multiplying by sqrt(radicand)
    lhs_multiplied = (diff_assumed * sqrt(radicand)).expand()
    rhs_multiplied = num.expand()

    # Equation: lhs_multiplied - rhs_multiplied = 0
    equation_to_solve = (lhs_multiplied - rhs_multiplied).expand()

    # Collect coefficients for powers of var
    collected_eq = equation_to_solve.as_poly(var)
    if collected_eq is None:
        return None

    # Get all coefficients (should be zero)
    coeff_equations = list(collected_eq.all_coeffs())

    # Solve for all unknowns: [C0, C1, ..., Cm, K]
    unknowns = list(coeffs) + [K]
    solutions = solve(coeff_equations, unknowns, dict=True)

    if not solutions or len(solutions) == 0:
        return None

    # Use the first solution
    sol = solutions[0]
    Q_solved = Q.subs(sol)  # type: ignore
    K_val = sol[K]  # type: ignore

    # Construct final result
    integral_sqrt_inv = Integral(1 / sqrt(radicand), var)
    result = Q_solved * sqrt(radicand) + K_val * integral_sqrt_inv

    # Generate coefficient display string
    coeff_values = [f"C_{i} = {latex(sol[c])}" for i, c in enumerate(coeffs)] + [f"K = {latex(K_val)}"]
    coeff_display = "所有系数均为零" if not coeff_values else rf",\;".join(coeff_values)

    var_latex, radicand_latex, expr_latex, result_latex = latex(
        var), latex(radicand), latex(expr), latex(result)

    explanation = (
        rf"形如$\int \frac{{P(x)}}{{\sqrt{{ax^2+bx+c}}}}dx\;$使用待定系数法:\;"
        rf"设原函数为\;$Q(x) \sqrt{{{radicand_latex}}} + K \int \frac{{1}}{{\sqrt{{{radicand_latex}}}}}dx$,\;"
        rf"其中\;$Q(x)$为\;{m}\;次多项式.\;通过求导并比较系数, 解得：\;${coeff_display}$,\;即\;"
        rf"$\int {expr_latex}\,d{var_latex} = {result_latex}$"
    )

    return result, explanation


def try_radical_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Attempt radical substitution for integrals containing nested radicals,
    e.g., sqrt(x), sqrt(x + sqrt(x)), (x + 1)^{2/3}, etc.

    Strategy:
      - Find a radical term r = (g(x))^{p/q} with -1 < p/q < 1.
      - Set u = r and g(x) = u^{q/p}, then solve for x if possible.
      - This implementation assumes the simplest case: r = x^{1/n} or r = (linear in x)^{1/n}.

    Returns the integrated result and explanation if successful.
    """
    # Exclude log(f(x)+sqrt(f(x)))
    if isinstance(expr, log) and isinstance(expr.args[0], Add):
        return None

    # Collect all power expressions that are proper fractional powers of x.
    candidates = [atom for atom in expr.atoms(Pow) if isinstance(
        atom.exp, Rational) and -1 < atom.exp < 1 and atom.base.has(var)]

    # Sort by depth (simplest first) – optional but improves success rate
    candidates.sort(key=lambda r: r.count_ops())

    x_of_u, u = None, None
    for rad in candidates:
        base, _exp = rad.base, rad.exp  # rad = base**exp
        q = _exp.q  # denominator
        p = _exp.p  # numerator

        # We set u = rad = base**(p/q), then base = u**(q/p)
        # To proceed, we need to express x in terms of u.
        # Solve base = u**(q/p) for x.
        try:
            u = step_gene.get_available_sym(var)
            equation = Eq(base, u ** (Rational(q, p)))
            sol = solve(equation, var)
            if not sol:
                continue
            x_of_u = sol[0]
            # When inverting a radical with a multi-branch solution (e.g. sqrt of a
            # quadratic), pick the branch consistent with the "assume variable positive"
            # convention: the real solution x(u) that grows positive as u -> +oo. This
            # returns the textbook antiderivative valid on the x>0 region instead of the
            # sign-flipped other branch (fixes sqrt(x**2+2*x) shipping -integrand for x>0).
            if len(sol) > 1:
                probes = []
                for s in sol:
                    try:
                        v = complex(s.subs(u, 10 ** 6).evalf())
                    except (TypeError, ValueError):
                        continue
                    if abs(v.imag) < 1e-9:
                        probes.append((v.real, s))
                if probes:
                    x_of_u = max(probes, key=lambda t: t[0])[1]
        except Exception:
            pass

        try:
            dx_du = diff(x_of_u, u)
            # Replace any exact match of 'rad' with u
            new_expr = expr.subs(rad, u)
            # Substitute: every occurrence of 'var' to x(u), and rad to u
            new_expr = new_expr.subs(var, x_of_u)
            if new_expr.has(var) or new_expr.has(rad):
                continue
            new_expr = simplify(new_expr * dx_du)
            # Use a temporary variable with positive real assumptions to aid radical simplification.
            _t = Symbol('t', real=True, positive=True)
            new_expr = simplify(new_expr.subs(u, _t)).subs(_t, u).replace(
                Abs, lambda arg: arg)

            final_result = Integral(new_expr, u)

            # Store substitution
            step_gene.subs_dict[u] = rad
            explanation = (
                f"根式代换：令 ${u.name} = {latex(rad)}$, 则由 ${latex(base)} = {u.name}^{{{latex(Rational(q, p))}}}$ 解得 "
                f"${var.name} = {latex(x_of_u)}$, "
                f"且 $d{var.name} = {latex(dx_du)}\\,d{u.name}$. 积分化为 "
                f"$\\int {latex(new_expr)}\\,d{u.name}$"
            )
            return final_result, explanation

        except Exception as e:
            print(f"Error in try_radical_substitution: {e}")
            continue

    return None


def _has_fractional_power(expr: Expr, sym: Symbol) -> bool:
    """True if any power of ``sym`` (or an expression containing it) has a non-integer
    rational exponent — i.e. the expression is not yet a rational function of ``sym``."""
    for atom in expr.atoms(Pow):
        if atom.base.has(sym) and isinstance(atom.exp, Rational) and not atom.exp.is_integer:
            return True
    return False


def try_rationalizing_substitution(expr: Expr, var: Symbol, step_gene: BaseStepGenerator) -> RuleFunctionReturn:
    """Rationalizing substitution for integrands that are rational functions of two or
    more *distinct* fractional powers of the variable, e.g. 1/(sqrt(x) + x**(1/3)).

    The single-radical case is left to ``try_radical_substitution``; here we set
    ``u = var**(1/L)`` with ``L`` the LCM of the exponents' denominators, so every
    fractional power becomes an integer power of ``u`` and the integrand turns into a
    rational function that the partial-fraction machinery can integrate.
    """
    pows = [a for a in expr.atoms(Pow)
            if isinstance(a.exp, Rational) and not a.exp.is_integer and a.base == var]
    if len(pows) < 2:
        return None

    L = math.lcm(*[a.exp.q for a in pows])
    if L <= 1:
        return None

    u = step_gene.get_available_sym(var)
    var_of_u = u ** L
    dvar_du = diff(var_of_u, u)
    new_expr = simplify(powdenest(together(expr.subs(var, var_of_u) * dvar_du), force=True))
    if new_expr.has(var) or _has_fractional_power(new_expr, u):
        return None

    step_gene.subs_dict[u] = var ** Rational(1, L)
    explanation = (
        f"根式有理化代换：被积函数含多个不同分数次幂，令 ${u.name} = {latex(var ** Rational(1, L))}$，"
        f"则 ${var.name} = {latex(var_of_u)}$，$d{var.name} = {latex(dvar_du)}\\,d{u.name}$，"
        f"积分化为有理函数 $\\int {latex(new_expr)}\\,d{u.name}$"
    )
    return Integral(new_expr, u), explanation
