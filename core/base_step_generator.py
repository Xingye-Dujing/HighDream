from abc import ABC
from typing import List, Tuple

from sympy import Expr, latex


class BaseStepGenerator(ABC):
    """Abstract base class for generating and formatting the step-by-step evaluation."""

    def __init__(self) -> None:
        self._steps: List[Expr] = []
        self._explanations: List[str] = []

    def reset(self) -> None:
        """Reset internal state to prepare for a new calculation.

        Clear all recorded steps and their corresponding explanations.
        """
        self._steps = []
        self._explanations = []

    def add_step(self, expr: Expr, explanation: str = "") -> None:
        """Append a new evaluation step with an optional explanatory message."""
        self._steps.append(expr)
        self._explanations.append(explanation)

    def get_steps(self) -> Tuple[List[Expr], List[str]]:
        """Return the evaluation steps and their descriptions.

        Returns:
            Tuple:
            - A list of symbolic expressions representing each evaluation step.
            - A list of strings describing each step.
        """
        return self._steps, self._explanations

    def to_latex(self) -> str:
        """Format steps as a LaTeX align environment.
           To render it in a Jupyter notebook, use: ``display(Math(latex_string))``."""
        if not self._steps:
            return r"\begin{align}\end{align}"

        lines = []
        for i, (step, explanation) in enumerate(zip(self._steps, self._explanations)):
            line = r"&=" + latex(step) if i > 0 else "&" + latex(step)
            if explanation:
                line += rf" \quad \text{{{explanation}}}"
            lines.append(line)

        body = r"\newline".join(lines)
        return rf"\begin{{align}}{body}\end{{align}}"
