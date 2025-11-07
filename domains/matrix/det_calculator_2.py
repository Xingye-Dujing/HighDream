from collections import deque
from typing import Any, Deque, Dict, Tuple

from IPython.display import Math, display
from sympy import Expr, Matrix, latex, sympify, Determinant, ImmutableMatrix
from domains.differentiation import RULE_DICT, MATCHER_LIST
from core import BaseCalculator
from domains.matrix import (
    DetStepGenerator,
    zero_row_matcher, zero_row_rule,
    zero_column_matcher, zero_column_rule,
    duplicate_row_matcher, duplicate_row_rule,
    duplicate_column_matcher, duplicate_column_rule,
    diagonal_matcher, diagonal_rule,
    triangular_matcher, triangular_rule,
    scalar_multiple_row_matcher, scalar_multiple_row_rule,
    scalar_multiple_column_matcher, scalar_multiple_column_rule,
    linear_combination_matcher, linear_combination_rule,
    laplace_expansion_matcher, laplace_expansion_rule,
    vandermonde_rule, vandermonde_matcher,
    circulant_rule, circulant_matcher,
    symmetric_rule, symmetric_matcher,
)


class DetCalculator(BaseCalculator):
    def __init__(self):
        self.on_linear_combination = True
        super().__init__()
        self.step_generator = DetStepGenerator()

    def init_key_property(self):
        self.operation: Operation = Determinant
        self.rule_dict: RuleDict = RULE_DICT
        self.matcher_list: MatcherList = MATCHER_LIST

    def _initialize_rules(self):
        # 规则映射表：规则名称 -> 规则函数
        rule_mapping = {
            'zero_row': zero_row_rule,
            'zero_column': zero_column_rule,
            'duplicate_row': duplicate_row_rule,
            'duplicate_column': duplicate_column_rule,
            'diagonal': diagonal_rule,
            'triangular': triangular_rule,
            'scalar_multiple_row': scalar_multiple_row_rule,
            'scalar_multiple_column': scalar_multiple_column_rule,
            'linear_combination': linear_combination_rule,
            'laplace_expansion': laplace_expansion_rule,
            'vandermonde': vandermonde_rule,
            'circulant': circulant_rule,
            'symmetric': symmetric_rule,
        }

        # 匹配器列表. 注意: 前面的匹配优先级大于后面
        if self.on_linear_combination:
            matchers = [
                zero_row_matcher,
                zero_column_matcher,
                duplicate_row_matcher,
                duplicate_column_matcher,
                scalar_multiple_row_matcher,
                scalar_multiple_column_matcher,
                diagonal_matcher,
                triangular_matcher,
                vandermonde_matcher,
                circulant_matcher,
                symmetric_matcher,
                linear_combination_matcher,
                laplace_expansion_matcher,
            ]
        else:
            matchers = [
                zero_row_matcher,
                zero_column_matcher,
                duplicate_row_matcher,
                duplicate_column_matcher,
                scalar_multiple_row_matcher,
                scalar_multiple_column_matcher,
                diagonal_matcher,
                triangular_matcher,
                vandermonde_matcher,
                circulant_matcher,
                symmetric_matcher,
                laplace_expansion_matcher,
            ]

        # 注册所有规则
        for name, rule in rule_mapping.items():
            self.rule_registry._register_rule(name, rule)

        # 注册所有匹配器
        for matcher in matchers:
            self.rule_registry._register_matcher(matcher)

    def _matrix_to_immutable(self, matrix: Matrix) -> ImmutableMatrix:
        """将矩阵转换为不可变矩阵用于缓存键"""
        return ImmutableMatrix(matrix)

    def _get_determinant(self, matrix: Matrix) -> Determinant:
        """使用缓存池缓存 Determinant 对象"""
        key = self._matrix_to_immutable(matrix)
        if key not in self.cache:
            self.cache[key] = Determinant(matrix)
        return self.cache[key]

    def _apply_rule(self, det_expr: Determinant) -> Tuple[Expr, str]:
        """应用行列式计算规则并返回结果和解释"""
        matrix = det_expr.args[0]
        context = {'matrix': matrix}

        for rule in self.rule_registry.get_rule(det_expr, context):
            result = rule(det_expr, context)
            if result:
                return result

        # 如果无法应用规则，则直接使用 sympy 计算
        return det_expr.doit(), f"直接计算行列式: ${latex(det_expr)}$"

    def _compute(self, matrix_input: Any) -> None:
        """计算矩阵的行列式, 并记录所有步骤"""
        self._reset_process()

        # 将输入转换为 Matrix 对象
        if isinstance(matrix_input, str):
            matrix = Matrix(sympify(matrix_input))
        elif isinstance(matrix_input, list):
            matrix = Matrix(matrix_input)
        else:
            matrix = matrix_input

        # 初始行列式表达式 -> Determinant 对象
        initial_det = self._get_determinant(matrix)
        self.step_generator.add_step(initial_det)

        # 使用队列进行广度优先搜索
        queue: Deque[Any] = deque([matrix])
        matrix_to_det: Dict[Any, Any] = {
            self._matrix_to_immutable(matrix): initial_det}
        seen = set([self._matrix_to_immutable(matrix)])

        while queue:
            current_matrix = queue.popleft()
            current_matrix_key = self._matrix_to_immutable(current_matrix)
            current_det = matrix_to_det.get(current_matrix_key)

            if current_matrix_key in self.processed or current_det is None:
                continue

            self.processed.add(current_matrix_key)

            # 应用行列式计算规则
            new_expr, explanation = self._apply_rule(current_det)

            # 替换所有出现该矩阵的地方
            for key in list(matrix_to_det.keys()):
                matrix_to_det[key] = matrix_to_det[key].subs(
                    Determinant(current_matrix), new_expr
                )

            # 更新当前步骤
            current_step = matrix_to_det[self._matrix_to_immutable(matrix)]
            self.step_generator.add_step(current_step, explanation)

            # 如果规则应用成功E
            if new_expr != current_det:
                # 如果新表达式仍然包含行列式，将其部分加入队列
                if isinstance(new_expr, Determinant):
                    sub_matrix = new_expr.args[0]
                    sub_matrix_key = self._matrix_to_immutable(sub_matrix)
                    if sub_matrix_key not in matrix_to_det and sub_matrix_key not in seen:
                        matrix_to_det[sub_matrix_key] = new_expr
                        seen.add(sub_matrix_key)
                        queue.append(sub_matrix)
                elif new_expr.has(Determinant):
                    # 查找所有子行列式表达式
                    for sub_det in new_expr.atoms(Determinant):
                        sub_matrix = sub_det.args[0]
                        sub_matrix_key = self._matrix_to_immutable(sub_matrix)
                        if sub_matrix_key not in matrix_to_det and sub_matrix_key not in seen:
                            matrix_to_det[sub_matrix_key] = sub_det
                            seen.add(sub_matrix_key)
                            queue.append(sub_matrix)

        # 最后一步：尝试简化表达式
        if self.step_generator.steps:
            final_step = self.step_generator.steps[-1]
            if isinstance(final_step, tuple) and len(final_step) == 2:
                final_expr = final_step[0]
            else:
                final_expr = final_step

            simplified_step = self._cached_simplify(final_expr)
            if simplified_step != final_expr:
                self.step_generator.add_step(simplified_step, "简化表达式")

    def compute_list(self, matrix_input: Any) -> tuple[list]:
        """计算行列式并返回 list 格式的推导过程"""
        self._compute(matrix_input)
        return self.step_generator.get_steps()

    def compute_latex(self, matrix_input: Any) -> str:
        """计算行列式并返回 Latex 格式的推导过程"""
        self._compute(matrix_input)
        return self.step_generator.get_latex()


# 使用示例
if __name__ == "__main__":
    calculator = DetCalculator()

    matrix_2x2 = [[1, 2], [3, 4]]
    latex_output = calculator.compute_latex(matrix_2x2)
    display(Math(latex_output))

    matrix_3x3 = [[1, 0, 0], [0, 4, 0], [0, 0, 6]]
    latex_output = calculator.compute_latex(matrix_3x3)
    display(Math(latex_output))
