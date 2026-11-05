from abc import ABC
from typing import Tuple

# from IPython.display import Math, display
from sympy import Expr, latex, pprint

from utils import Context, Operation, RuleContext
from .base_calculator import BaseCalculator


class SelectRuleCalculator(BaseCalculator, ABC):
    """A calculator that allows manual selection of rules for expression transformation."""

    def _apply_rule(self, expr: Expr, operation: Operation, **context: Context) -> Tuple[Expr, str]:
        """Apply rule to the expression and return result with explanation."""
        rule_context: RuleContext = self._get_context_dict(**context)

        applicable_rules_list = list(self._rule_registry.get_applicable_rules(
            expr, rule_context))

        # Jupyter Notebook: display(Math(latex(expr)))
        # Use the cached operation so Limit displays need its variable/point/direction;
        # ``operation(expr)`` drops them and crashes for Limit.
        try:
            pprint(self._get_cached_operation(expr, operation, **context), use_unicode=True)
        except Exception:  # pylint: disable=broad-exception-caught
            print(self._get_cached_operation(expr, operation, **context))
        if applicable_rules_list:
            print("以下规则可应用：")
            for i, rule in enumerate(applicable_rules_list):
                print(f"{i + 1}. {rule.__name__}")
            index = input("请输入规则编号：")
            try:
                idx = int(index) - 1
            except ValueError:
                idx = -1
            if 0 <= idx < len(applicable_rules_list):
                rule = applicable_rules_list[idx]
                result = rule(expr, rule_context)
                if result:
                    # display(Math(latex(result[0])))
                    try:
                        pprint(result[0], use_unicode=True)
                    except Exception:  # pylint: disable=broad-exception-caught
                        print(result[0])
                    print(result[1])
                    return result
                print("此规则无法应用，请重新选择.")
            else:
                print("编号无效，使用 SymPy 回退。")
        else:
            print("没有可应用的规则，使用 SymPy 回退。")

        # Fallback to SymPy if no rule matches
        operation_obj = self._perform_operation(expr, operation, **context)
        return operation_obj.doit(), f"需手动计算表达式: ${latex(operation_obj)}$"
