from sympy import Matrix, latex, simplify

from core import BaseStepGenerator


class RefStepGenerator(BaseStepGenerator):

    def get_latex(self, steps, explanations):
        """生成适合矩阵的 Latex 格式(支持分支标题等)"""
        latex_str = "\\begin{align}\n"
        for i, (step, explanation) in enumerate(zip(steps, explanations)):
            # step 可能是字符串(分支标题或说明)或 Matrix
            if isinstance(step, str):
                # 作为整行说明
                step_str = f"& \\text{{{step}}} \\quad & \\text{{{explanation}}}"
            else:
                # 处理矩阵或一般表达式
                try:
                    if isinstance(step, Matrix):
                        m_latex = self._matrix_to_latex(step)
                    else:
                        m_latex = latex(step)
                except Exception:
                    m_latex = str(step)
                # 如果说明包含 "合并/分支" 之类字样, 直接显示矩阵与说明
                if i == 0:
                    step_str = f"& {m_latex} \\quad & \\text{{{explanation}}}"
                elif '分支' in explanation or '合并' in explanation or '条件' in explanation:
                    step_str = f"& {m_latex} \\quad & \\text{{{explanation}}}"
                else:
                    step_str = f"&\\Rightarrow {m_latex} \\quad & \\text{{{explanation}}}"
            latex_str += step_str
            if i < len(steps) - 1:
                latex_str += "\\\\\n"
        latex_str += "\\end{align}"
        return latex_str

    def _matrix_to_latex(self, matrix) -> str:
        simplified_matrix = matrix.applyfunc(
            lambda x: simplify(x) if x != 0 else 0)
        return latex(simplified_matrix)
