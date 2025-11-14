from sympy import latex
from core import BaseStepGenerator


class DetStepGenerator(BaseStepGenerator):
    def get_latex(self) -> str:
        """生成 Latex 格式的推导过程"""
        latex_str = "\\begin{align}"

        # 匹配对应的变换和原理
        for i, (step, explanation) in enumerate(zip(self.steps, self._explanations)):
            if i == 0:
                latex_str += '&' + latex(step)
            else:
                # 对齐符号
                latex_str += "\\\\ &=" + latex(step)

            if explanation:
                latex_str += f" \\quad \\text{{{explanation}}}"
            latex_str += "\\\\\n"

        latex_str += "\\end{align}"
        return latex_str
