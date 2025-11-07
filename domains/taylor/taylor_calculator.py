# TODO 结果有问题
# TODO 规则返回的 latex 渲染有问题

from collections import deque
from sympy import sympify, Symbol, series, latex, simplify, sin, cos
from core.base_calculator import BaseCalculator
from .rules.function_rules import (
    const_rule, var_rule, sin_rule, cos_rule, exp_rule, log_rule,
    add_rule, mul_rule, pow_rule, composite_rule,
    const_matcher, var_matcher, sin_matcher, cos_matcher, exp_matcher, log_matcher,
    add_matcher, mul_matcher, pow_matcher, composite_matcher
)


class TaylorCalculator(BaseCalculator):
    def _initialize_rules(self):
        # 注册泰勒展开规则
        self.rule_registry.register_rule('const', const_rule)
        self.rule_registry.register_rule('var', var_rule)
        self.rule_registry.register_rule('sin', sin_rule)
        self.rule_registry.register_rule('cos', cos_rule)
        self.rule_registry.register_rule('exp', exp_rule)
        self.rule_registry.register_rule('log', log_rule)
        self.rule_registry.register_rule('add', add_rule)
        self.rule_registry.register_rule('mul', mul_rule)
        self.rule_registry.register_rule('pow', pow_rule)
        self.rule_registry.register_rule('composite', composite_rule)

        # 注册规则匹配器
        self.rule_registry.register_matcher(const_matcher)
        self.rule_registry.register_matcher(var_matcher)
        self.rule_registry.register_matcher(sin_matcher)
        self.rule_registry.register_matcher(cos_matcher)
        self.rule_registry.register_matcher(exp_matcher)
        self.rule_registry.register_matcher(log_matcher)
        self.rule_registry.register_matcher(add_matcher)
        self.rule_registry.register_matcher(mul_matcher)
        self.rule_registry.register_matcher(pow_matcher)
        self.rule_registry.register_matcher(composite_matcher)

    def _apply_rule(self, expr, var, point, order):
        """应用泰勒展开规则并返回结果和解释"""
        context = {'variable': var, 'point': point, 'order': order}
        
        for rule in self.rule_registry.get_rule(expr, context):
            result = rule(expr, context)
            if result:
                return result

        # 默认使用 SymPy 的 series 函数
        taylor_series = series(expr, var, point, order + 1).removeO()
        return taylor_series, f"使用\,SymPy\,计算泰勒展开: ${latex(taylor_series)} + O(({latex(var)}-{latex(point)})^{{{order+1}}})$"

    def _compute(self, expr, var, point, order):
        """计算表达式的泰勒展开，并记录所有步骤"""
        expr = sympify(expr)
        self._reset_process()

        # 初始表达式
        self.step_generator.add_step(
            expr, f"计算泰勒展开: 在\,${latex(var)} = {latex(point)}$\,处展开到阶$\,{order}$")

        # 预处理表达式
        simple_expr = simplify(expr)
        if simple_expr != expr:
            expr = simple_expr
            self.step_generator.add_step(
                expr, "简化表达式")

        # 使用队列进行广度优先搜索
        queue = deque([expr])
        expr_to_taylor = {expr: expr}

        while queue:
            current_expr = queue.popleft()
            current_taylor = expr_to_taylor[current_expr]

            # 如果已经处理过，跳过
            if current_expr in self.processed:
                continue

            self.processed.add(current_expr)

            # 提取表达式中的唯一变量: 换元法的更好实现
            try:
                expr_var = next(iter(current_expr.free_symbols))
            except:
                expr_var = Symbol('x')

            # 应用泰勒展开规则
            new_expr, explanation = self._apply_rule(current_expr, expr_var, point, order)
            # 替换所有包含该子表达式要展开的部分
            for key in list(expr_to_taylor.keys()):
               expr_to_taylor[key] = expr_to_taylor[key].subs(
                    series(current_expr, var, point,
                          order), new_expr
                )
            # 更新当前步骤
            current_step = expr_to_taylor[expr]
            self.step_generator.add_step(current_step, explanation)

            # 如果规则应用成功
            if new_expr != current_taylor:
                # 如果新表达式仍然包含需要进一步展开的部分，将其加入队列
                if new_expr.has(series):
                    # 查找所有需要进一步展开的子表达式
                    for sub_expr in new_expr.atoms(series):
                        if sub_expr not in expr_to_taylor:
                            expr_to_taylor[sub_expr] = sub_expr
                            queue.append(sub_expr)
                elif new_expr.has(Symbol):
                    # 查找所有包含变量的子表达式
                    for sub_expr in new_expr.atoms(Symbol):
                        if sub_expr != var and sub_expr not in expr_to_taylor:
                            expr_to_taylor[sub_expr] = series(
                                sub_expr, var, point, order + 1)
                            queue.append(sub_expr)

        # 最后一步：尝试简化表达式
        if self.step_generator.steps:
            final_step = self.step_generator.steps[-1]
            simplified_step = simplify(final_step.removeO()) if hasattr(
                final_step, 'removeO') else simplify(final_step)
            # 如果简化后的表达式与原表达式不同，则添加简化步骤
            if simplified_step != final_step:
                self.step_generator.add_step(simplified_step, "简化泰勒展开式")

    def compute_list(self, expr, var=Symbol('x'), point=0, order=5):
        """计算泰勒展开并返回步骤列表"""
        self._compute(expr, var, point, order)
        return self.step_generator.get_steps()

    def compute_latex(self, expr, var=Symbol('x'), point=0, order=5):
        """计算泰勒展开并返回LaTeX格式的推导过程"""
        self._compute(expr, var, point, order)
        return self.step_generator.get_latex()


# 使用示例
if __name__ == "__main__":
    x = Symbol('x')

    calculator = TaylorCalculator()
    latex_output = calculator.compute_latex(sin(x) * cos(x), x, 0, 5)

    print("泰勒展开步骤:")
    print(latex_output)

    # 直接计算泰勒展开
    series_result = calculator.compute_series(sin(x) * cos(x), x, 0, 5)
    print(f"\n直接计算结果: {series_result}")
