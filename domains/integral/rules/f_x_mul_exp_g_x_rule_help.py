from sympy import Add, Expr, Integral, diff, exp, latex, integrate, simplify
from utils import RuleContext, RuleFunctionReturn, is_elementary_expression


def handle_fx_mul_exp_gx(expr: Expr, exp_term: exp, another_term: Expr, context: RuleContext) -> RuleFunctionReturn:
    var = context['variable']
    step_gene = context['step_generator']

    integrate_result = simplify(integrate(expr, var))
    if not is_elementary_expression(integrate_result):
        return None
    diff_g_x = simplify(diff(exp_term.args[0], var))
    m_x = simplify(integrate_result/exp_term)
    diff_m_x = simplify(diff(m_x, var))
    f_x = simplify(diff_m_x + diff_g_x * m_x)
    add_term = simplify((another_term - f_x)*exp_term)
    if add_term.equals(0):
        result = simplify(m_x*exp_term)
    else:
        return None

    step_gene.add_step('None', '')
    step_gene.add_step(
        'None', f'$\\int {latex(expr)}\\,\\mathrm{{d}} {var}$')
    step_gene.add_step(
        'None', f'$= \\int {latex(diff_m_x*exp_term)}\\,\\mathrm{{d}} {var} + \\int {latex((another_term-diff_m_x)*exp_term)}\\,\\mathrm{{d}} {var}$')
    step_gene.add_step(
        'None', f'$= \\int {latex(diff_m_x*exp_term)}\\,\\mathrm{{d}} {var} + \\int {latex(m_x)}\\,\\mathrm{{d}} {latex(exp_term)}$')
    step_gene.add_step(
        'None', f'$= \\int {latex(diff_m_x*exp_term)}\\,\\mathrm{{d}} {var} + {latex(result)} - \\int {latex(diff_m_x*exp_term)}\\,\\mathrm{{d}} {var}$')
    step_gene.add_step(
        'None', f'$= {latex(result)} + C$')
    step_gene.add_step('None', '')

    mid_process = Integral((diff_m_x+m_x*diff_g_x)*exp_term, var)
    return result, (rf'$\int f({var})\,e^{{g({var})}}\,d{var} \to 变换表达式凑积的微分形式:'
                    rf'\int {latex(expr)}\, d{var}= {latex(mid_process)} ='
                    rf"\int (m'({var})+m({var})\,g'({var}))\,e^{{g({var})}}\,d{var} = "
                    rf'm({var})\,e^{{g({var})}},\,'
                    rf'此处\, m({var}) = {latex(m_x)},\,g({var}) = {latex(exp_term.args[0])}$')


def special_add_split_exp_term(expr: Expr, context: RuleContext) -> RuleFunctionReturn:
    expr_copy = expr
    expr = expr.expand()
    if not isinstance(expr, Add):
        return None

    exp_term = []
    another_term = []
    for arg in expr.args:
        if arg.has(exp):
            exp_term.append(arg)
        else:
            another_term.append(arg)

    if not another_term:
        return None

    var = context['variable']
    exp_term_add = simplify(Add(*exp_term))
    another_term_add = simplify(Add(*another_term))
    result = Integral(exp_term_add, var)+Integral(another_term_add, var)
    return result, rf'应用加法规则: $\int {latex(expr_copy)}\,d {latex(var)} = {latex(result)}$'
