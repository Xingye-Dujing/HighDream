from sympy import (
    Dummy, Expr, Integral, Mul, Pow, Symbol, Wild, acos, asin, atan, cos,
    diff, exp, integrate, latex, log, sec, simplify, sin, sqrt, tan
)

from utils import Context, MatcherFunctionReturn, RuleFunctionReturn
from utils.latex_formatter import wrap_latex
from domains.integral import select_parts_u_dv


def parts_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """
    分部积分法：∫u dv = uv - ∫v du
    基于 LIATE 法则: 智能选择 u 和 dv。
    """
    var = context['variable']
    if isinstance(expr, Mul) and len(expr.args) == 2:
        u, dv = select_parts_u_dv(expr, var)
        du = diff(u, var)
        v = integrate(dv, var)
        result = u * v - integrate(v * du, var)
        return result, f"分部积分法 (LIATE 选择 $u={latex(u)}$, $dv={latex(dv)}\\,d{var}$): $\\int {latex(expr)}\\,d{var} = {latex(u)}\\cdot{latex(v)} - \\int {latex(v)}\\cdot{latex(du)}\\,d{var} + C$"
    return None


def substitution_rule(expr: Expr, context: Context) -> RuleFunctionReturn:
    """换元法：支持复合函数、三角代换、根式代换、指数/对数代换"""
    var = context['variable']
    u = Dummy('u')  # 避免变量冲突
    # 策略 1：标准形式 f(g(x)) * g'(x)
    result = _try_standard_substitution(expr, var, u)
    if result:
        return result
    # 策略 2：三角代换
    result = _try_trig_substitution(expr, var, u)
    if result:
        return result
    # 策略 3：根式代换
    result = _try_radical_substitution(expr, var, u)
    if result:
        return result
    # 策略 4：指数/对数代换
    result = _try_exp_log_substitution(expr, var, u)
    if result:
        return result

    return None


def _try_standard_substitution(expr: Expr, var: Symbol, u: Symbol) -> RuleFunctionReturn:
    """匹配 ∫f(g(x)) * g'(x) dx 形式，支持 exp(sin(x))*cos(x) 等嵌套函数"""
    # 如果表达式不是乘法，尝试直接积分 f(g(x)) * g'(x)
    if not isinstance(expr, Mul):
        factors = [expr]
    else:
        factors = expr.args
    # 查找所有形如 f(g(x)) 的函数嵌套
    for factor in factors:
        if factor.is_Function and len(factor.args) == 1:
            inner_expr = factor.args[0]  # g(x)
            if not inner_expr.has(var):
                continue
            # 计算 g'(x)
            gp = diff(inner_expr, var)
            if gp.is_zero:
                continue
            # 检查 g'(x) 是否在表达式中作为因子出现
            # 即：expr / f(g(x)) 是否包含 g'(x) 或其倍数
            try:
                outer_part = expr / factor  # 剩余部分
                if outer_part.is_zero:
                    continue
                # 检查 outer_part 是否与 g'(x) 成比例
                ratio = outer_part / gp
                if ratio.is_constant():
                    # f(u) = factor.subs(inner_expr, u)
                    f_u = factor.func(u)  # exp(u), cos(u), etc.
                    inner_integral = integrate(f_u, u)
                    if inner_integral.has(Integral):
                        continue
                    result = inner_integral.subs(u, inner_expr)
                    explanation = (
                        f"换元法: 令 $u = {latex(inner_expr)}$, $du = {latex(gp)}\\,d{var}$, "
                        f"原式化为 $\\int {latex(f_u)}\\,du = {latex(inner_integral)}$, "
                        f"故积分为 ${latex(result)} + C$"
                    )
                    return result, explanation
            except Exception:
                continue
    return None


def _try_trig_substitution(expr: Expr, var: Symbol, _u: Symbol) -> RuleFunctionReturn:
    """处理 √(a² - x²), √(a² + x²), √(x² - a²) 的三角代换"""

    a = Wild('a', exclude=[var])
    # 1. √(a² - x²) → x = a*sin(θ)
    match = expr.find(sqrt(a**2 - var**2))
    if match:
        a_val = match[0]
        theta = Dummy('theta')
        x_sub = a_val * sin(theta)
        dx_dtheta = diff(x_sub, theta)
        new_expr = expr.subs(var, x_sub) * dx_dtheta
        int_theta = integrate(new_expr, theta)
        if int_theta.has(Integral):
            return None
        result = int_theta.subs(theta, asin(var / a_val))
        explanation = (
            f"三角代换: $\\sqrt{{{latex(a_val)}^2 - {var}^2}}$ 形式, "
            f"令 ${var} = {latex(a_val)} \\sin\\theta$, "
            f"得 $\\int ... d\\theta = {latex(int_theta)}$, "
            f"回代后为 ${latex(result)} + C$"
        )
        return result, explanation
    # 2. √(a² + x²) → x = a*tan(θ)
    match = expr.find(sqrt(a**2 + var**2))
    if match:
        a_val = match[0]
        theta = Dummy('theta')
        x_sub = a_val * tan(theta)
        dx_dtheta = diff(x_sub, theta)
        new_expr = expr.subs(var, x_sub) * dx_dtheta
        int_theta = integrate(new_expr, theta)
        if int_theta.has(Integral):
            return None
        result = int_theta.subs(theta, atan(var / a_val))
        explanation = (
            f"三角代换: $\\sqrt{{{latex(a_val)}^2 + {var}^2}}$ 形式, "
            f"令 ${var} = {latex(a_val)} \\tan\\theta$, 积分得 ${latex(result)} + C$"
        )
        return result, explanation
    # 3. √(x² - a²) → x = a*sec(θ)
    match = expr.find(sqrt(var**2 - a**2))
    if match:
        a_val = match[0]
        theta = Dummy('theta')
        x_sub = a_val * sec(theta)
        dx_dtheta = diff(x_sub, theta)
        new_expr = expr.subs(var, x_sub) * dx_dtheta
        int_theta = integrate(new_expr, theta)
        if int_theta.has(Integral):
            return None
        result = int_theta.subs(theta, acos(a_val / var))
        explanation = (
            f"三角代换: $\\sqrt{{{var}^2 - {latex(a_val)}^2}}$ 形式, "
            f"令 ${var} = {latex(a_val)} \\sec\\theta$, 积分得 ${latex(result)} + C$"
        )
        return result, explanation

    return None


def _try_radical_substitution(expr: Expr, var: Symbol, u: Symbol) -> RuleFunctionReturn:
    """根式代换：如 √x, √(x + √x)"""
    radicals = expr.atoms(Pow)
    for rad in radicals:
        if rad.exp.is_Rational and rad.exp < 1 and var in rad.free_symbols:
            n = int(1 / rad.exp)  # 如 1/2 → n=2
            inner_expr = rad.base  # 如 x, x + √x
            # 令 u = rad，即 u = √x → x = u²
            x_sub = inner_expr.subs(var, u**n)
            try:
                dx_du = diff(x_sub, u)
                # 新表达式 = 原式替换 x → x(u)，并乘以 dx/du
                new_expr = expr.subs(rad, u).subs(var, x_sub) * dx_du
                result_expr = integrate(new_expr, u)
                if result_expr.has(Integral):
                    continue
                final_result = result_expr.subs(u, rad)
                explanation = (
                    f"根式代换: 令 $u = {latex(rad)}$, 则 ${var} = {latex(x_sub)}$, "
                    f"$d{var} = {latex(dx_du)}\\,du$, 积分得 ${latex(final_result)} + C$"
                )
                return final_result, explanation
            except Exception:
                continue
    return None


def _try_exp_log_substitution(expr: Expr, var: Symbol, u: Symbol) -> RuleFunctionReturn:
    """指数/对数代换"""
    # e^x 代换
    if expr.has(exp(var)):
        x_sub = log(u)
        dx_du = diff(x_sub, u)
        new_expr = expr.subs(exp(var), u).subs(var, x_sub) * dx_du
        result_expr = integrate(new_expr, u)
        if not result_expr.has(Integral):
            final_result = result_expr.subs(u, exp(var))
            explanation = (
                f"指数代换: 令 $u = e^{{{var}}}$, ${var} = \\ln u$, "
                f"$d{var} = \\frac{{1}}{{u}}\\,du$, 积分得 ${latex(final_result)} + C$"
            )
            return final_result, explanation
    # ln(x) 代换
    if expr.has(log(var)):
        x_sub = exp(u)
        dx_du = diff(x_sub, u)
        new_expr = expr.subs(log(var), u).subs(var, x_sub) * dx_du
        result_expr = integrate(new_expr, u)
        if not result_expr.has(Integral):
            final_result = result_expr.subs(u, log(var))
            explanation = (
                f"对数代换: 令 $u = \\ln {var}$, ${var} = e^u$, "
                f"$d{var} = e^u\\,du$, 积分得 ${latex(final_result)} + C$"
            )
            return final_result, explanation

    return None


def parts_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    var = _context['variable']
    if isinstance(expr, Mul) and len(expr.args) == 2:
        has_log = any(isinstance(arg, log) for arg in expr.args)
        has_poly = any(arg.is_polynomial(var) for arg in expr.args)
        has_trig = any(arg.has(sin, cos) for arg in expr.args)
        has_exp = any(isinstance(arg, exp) for arg in expr.args)
        if has_log or (has_poly and (has_trig or has_exp)):
            return 'parts'
    return None


def substitution_matcher(expr: Expr, _context: Context) -> MatcherFunctionReturn:
    """换元法匹配器：检测是否适合代换"""
    var = _context['variable']
    # 1. 标准形式 f(g(x)) * g'(x)
    # 快速排除常数
    if not expr.has(var):
        return None
    # 获取所有乘法因子
    if expr.is_Mul:
        factors = expr.args
    else:
        factors = [expr]
    # 查找形如 f(g(x)) 的函数嵌套
    for factor in factors:
        if factor.is_Function and len(factor.args) == 1:
            inner = factor.args[0]
            if not inner.has(var):
                continue
            gp = diff(inner, var)
            if gp.is_zero:
                continue
            # 计算 expr / factor → 剩余部分
            outer_part = expr / factor
            if outer_part.is_zero:
                continue
            # 检查 outer_part 是否与 gp 成比例（即存在常数 k 使得 outer_part = k * gp）
            if outer_part.is_constant():
                # 特殊情况：f(g(x)) 本身是唯一因子，如 ∫sin(ln(x)) dx（不匹配）
                continue
            # 简化 ratio = outer_part / gp
            ratio = simplify(outer_part / gp)
            if ratio.is_constant():
                return 'substitution'
    # 2. 三角代换特征
    if (expr.find(sqrt(1 - var**2)) or
        expr.find(sqrt(1 + var**2)) or
            expr.find(sqrt(var**2 - 1))):
        return 'substitution'
    # 3. 根式
    if any(isinstance(arg, Pow) and arg.exp.is_Rational and arg.exp < 1 for arg in expr.args):
        return 'substitution'
    # 4. 指数/对数
    if expr.has(exp(var)) or expr.has(log(var)):
        return 'substitution'
    return None
