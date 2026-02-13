from sympy import latex

from core import BaseStepGenerator


class DetStepGenerator(BaseStepGenerator):
    def get_latex(self) -> str:
        """Generate the derivation process in LaTeX format.

        This method converts the stored steps and explanations into a properly
        formatted LaTeX align environment for mathematical expressions.

        Returns:
            str: A LaTeX formatted string containing all steps with explanations.
        """
        latex_str = "\\begin{align}"

        # Match corresponding transformations and principles
        for i, (step, explanation) in enumerate(zip(self.steps, self._explanations)):
            if i == 0:
                latex_str += '&' + latex(step)
            else:
                # Alignment symbol
                latex_str += "\\\\ &=" + latex(step)

            if explanation:
                latex_str += f" \\quad \\text{{{explanation}}}"
            latex_str += "\\\\\n"

        latex_str += "\\end{align}"
        return latex_str
