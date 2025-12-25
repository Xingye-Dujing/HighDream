from typing import Tuple
from IPython.display import Math, display
from sympy import Expr, latex

from utils import Context, Operation, RuleContext
from .base_calculator import BaseCalculator


class SelectRuleCalculator(BaseCalculator):
    """A calculator that allows manual selection of rules for expression transformation."""

    def _apply_rule(self, expr: Expr, operation: Operation, **context: Context) -> Tuple[Expr, str]:
        """Apply rule to the expression and return result with explanation."""
        rule_context: RuleContext = self._get_context_dict(**context)

        applicable_rules_list = self._rule_registry.get_applicable_rules(
            expr, rule_context)
        for rule in applicable_rules_list[:]:
            if not self._check_rule_is_can_apply(rule):
                applicable_rules_list.remove(rule)

        display(Math(latex(expr)))
        print("以下规则可应用：")
        for i, rule in enumerate(applicable_rules_list):
            print(f"{i+1}. {rule.__name__}")
        index = input("请输入规则编号：")
        rule = applicable_rules_list[int(index)-1]
        result = rule(expr, rule_context)

        if result:
            display(Math(latex(result[0])))
            return result

        print("此规则无法应用，请重新选择.")

        # Fallback to SymPy if no rule matches
        operation_obj = self._perform_operation(expr, operation, **context)
        return operation_obj.doit(), f"需手动计算表达式: ${latex(operation_obj)}$"
