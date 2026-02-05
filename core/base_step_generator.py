from abc import ABC
from typing import List, Tuple

from sympy import Dummy, Expr, Symbol, latex


available_sym_chars = [
    'u', 'v', 'x', 'y', 'h', 'z', 't', 's', 'r', 'q', 'w',
    'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'g', 'f', 'e',
    'd', 'b', 'a', 'c'
]


class BaseStepGenerator(ABC):
    """Abstract base class for generating and formatting the step-by-step evaluation."""

    def __init__(self) -> None:
        self.steps: List[Expr] = []
        self._explanations: List[str] = []
        # The keys are post-substitution variables and values are pre-substitution expressions.
        # key: new symbol, value: original symbol
        self.subs_dict: dict = {}
        # copy() to avoid modifying the original list
        self.available_sym_chars = available_sym_chars.copy()

    def reset(self) -> None:
        """Reset internal state to prepare for a new calculation.

        Clear all recorded steps and their corresponding explanations.
        """
        self.steps = []
        self._explanations = []
        self.subs_dict = {}
        self.available_sym_chars = available_sym_chars.copy()

    def get_available_sym(self, var: Symbol) -> Symbol:
        if var.name in self.available_sym_chars:
            self.available_sym_chars.remove(var.name)
        if not self.available_sym_chars:
            return Dummy(var.name)
        new_sym_char = self.available_sym_chars[0]
        self.available_sym_chars.remove(new_sym_char)
        return Symbol(new_sym_char)

    def add_step(self, expr: Expr, explanation: str = "") -> None:
        """Append a new evaluation step with an optional explanatory message."""
        self.steps.append(expr)
        self._explanations.append(explanation)

    def get_steps(self) -> Tuple[List[Expr], List[str]]:
        """Return the evaluation steps and their descriptions.

        Returns:
            Tuple:
            - A list of symbolic expressions representing each evaluation step.
            - A list of strings describing each step.
        """
        return self.steps, self._explanations

    def to_latex(self) -> str:
        """Format steps as a LaTeX align environment.
           To render it in a Jupyter notebook, use: ``display(Math(latex_string))``."""
        if not self.steps:
            return r"\begin{align}\end{align}"

        lines = []
        for i, (step, explanation) in enumerate(zip(self.steps, self._explanations)):
            if step == 'None':
                line = rf"& \quad \quad"
            else:
                line = r"&=" + latex(step) if i > 0 else "&" + latex(step)
            if explanation:
                line += rf" \quad \text{{{explanation}}}"
            lines.append(line)

        body = r"\newline".join(lines)
        return rf"\begin{{align}}{body}\end{{align}}"
