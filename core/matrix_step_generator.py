from typing import List


class MatrixStepGenerator():
    """Generate and manage step-by-step mathematical operations in LaTeX format.

    This class provides functionality to collect individual calculation steps
    and output them as a formatted LaTeX align environment.
    """

    def __init__(self) -> None:
        """Initialize an empty list to store calculation steps."""
        self.steps: List[str] = []

    def clear(self) -> None:
        """Clear all stored calculation steps."""
        self.steps = []

    def add_step(self, step: str) -> None:
        """Add a single calculation step to the collection.

        Args:
            step (str): A LaTeX-formatted string representing one calculation step.
        """
        self.steps.append(step)

    def get_steps_latex(self) -> str:
        """Generate a LaTeX align environment containing all calculation steps.

        Returns:
            str: A LaTeX-formatted string with all steps in an align environment,
                 with each step on a new line.
        """
        latex_str = "\\begin{align}"
        for step in self.steps:
            step_str = f"& {step}"
            latex_str += step_str
            # Add line break for next step
            latex_str += r"\\"
        latex_str += "\\end{align}"
        return latex_str
