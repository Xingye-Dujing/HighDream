from sympy import latex

from core import BaseStepGenerator
from core.base_step_generator import format_note


class DetStepGenerator(BaseStepGenerator):
    def get_latex(self) -> str:
        """Generate the derivation process in LaTeX format.

        This method converts the stored steps and explanations into a properly
        formatted LaTeX align environment for mathematical expressions.

        Returns:
            str: A LaTeX formatted string containing all steps with explanations.
        """
        lines = []
        for i, (step, explanation) in enumerate(zip(self.steps, self._explanations)):
            if i == 0:
                line = '&' + latex(step)
            else:
                line = "&= " + latex(step)
            if explanation:
                line += " \\quad " + format_note(explanation)
            lines.append(line)
        return r"\begin{align}" + " \\\\ ".join(lines) + r"\end{align}"
