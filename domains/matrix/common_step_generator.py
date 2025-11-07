from typing import List


class CommonStepGenerator():
    def __init__(self):
        self.steps = []

    def clear(self):
        self.steps = []

    def add_step(self, step: str) -> None:
        self.steps.append(step)

    def get_steps_latex(self) -> List[str]:
        latex_str = "\\begin{align}"
        for step in self.steps:
            step_str = f"& {step}"
            latex_str += step_str
            # 换行并空一行
            latex_str += r"\\\\"
        latex_str += "\\end{align}"
        return latex_str
