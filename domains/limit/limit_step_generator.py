from core import BaseStepGenerator
from core.base_step_generator import format_note
from utils.cache_utils import cached_latex


class LimitStepGenerator(BaseStepGenerator):

    def to_latex_both(self) -> str:
        """Generate a LaTeX string representing two-sided limits' steps."""
        if not self.steps:
            return r"\begin{align}\end{align}"

        lines = []
        for i, (step, explanation) in enumerate(zip(self.steps, self._explanations)):
            step_type = self._get_step_type(explanation)

            if step_type == "left_limit_start":
                line = rf"& {cached_latex(step)} \quad {format_note(explanation)}"
            elif step_type == "right_limit_start":
                line = rf"\newline \newline & {cached_latex(step)} \quad {format_note(explanation)}"
            elif step_type == "comparison":
                line = rf"\newline \newline & {format_note(explanation)}"
            else:
                line = r"&=" + cached_latex(step) if i > 0 else "&" + cached_latex(step)
                if explanation:
                    line += r"\quad " + format_note(explanation)
            lines.append(line)

        body = r"\newline".join(lines)
        return rf"\begin{{align}}{body}\end{{align}}"

    @staticmethod
    def _get_step_type(explanation: str) -> str:
        """Determine the step type based on keywords in the explanation."""
        if "计算左极限" in explanation:
            return "left_limit_start"
        if "计算右极限" in explanation:
            return "right_limit_start"
        if "左极限" in explanation and "右极限" in explanation:
            return "comparison"
        return "normal"
