from abc import ABC
from typing import List, Tuple

from sympy import Expr, Symbol

from utils.cache_utils import cached_latex

# from sympy import Dummy


available_sym_chars = [
    'u', 'v', 'x', 'y', 'h', 'z', 't', 's', 'r', 'q', 'w',
    'p', 'o', 'n', 'm', 'l', 'k', 'j', 'i', 'g', 'f', 'e',
    'd', 'b', 'a', 'c'
]


def format_note(explanation) -> str:
    """Render a step explanation as a MathJax fragment.

    Text between ``$...$`` becomes live LaTeX (so formulas render); the rest is
    wrapped in ``\\text{}`` as prose. This avoids showing literal ``$`` or unicode
    math symbols as plain text.
    """
    parts = str(explanation).split('$')
    out = []
    for i, seg in enumerate(parts):
        if seg == '':
            continue
        out.append(seg if i % 2 == 1 else r"\text{" + seg + r"}")
    return ''.join(out)


class BaseStepGenerator(ABC):
    """Abstract base class for generating and formatting the step-by-step evaluation."""

    def __init__(self) -> None:
        self.steps: List[Expr] = []
        self._explanations: List[str] = []
        # The keys are post-substitution variables and values are pre-substitution expressions.
        # key: new symbol, value: the original symbol.
        self.subs_dict: dict = {}
        # copy() to avoid modifying the original list
        self.available_sym_chars = available_sym_chars.copy()

    def reset(self) -> None:
        """Reset internal state to prepare for a new calculation.
        Clear all recorded steps and their corresponding explanations."""

        self.steps = []
        self._explanations = []
        self.subs_dict = {}
        self.available_sym_chars = available_sym_chars.copy()

    def get_available_sym(self, var: Symbol, **symbol_kwargs) -> Symbol:
        if var.name in self.available_sym_chars:
            self.available_sym_chars.remove(var.name)
        if not self.available_sym_chars:
            raise ValueError("No more available symbols")
            # return Dummy(var.name)
        new_sym_char = self.available_sym_chars[0]
        self.available_sym_chars.remove(new_sym_char)
        return Symbol(new_sym_char, **symbol_kwargs)

    def add_step(self, expr: List[Expr | None], explanation: str = "") -> None:
        """Append a new evaluation step with an optional explanatory message."""
        self.steps.append(expr)
        self._explanations.append(explanation)

    def get_steps(self) -> Tuple[List[Expr], List[str]]:
        """Return the evaluation steps and their descriptions.

        Returns:
            Tuple:
            - A list of symbolic expressions representing each evaluation step.
            - A list of strings describing each step."""

        return self.steps, self._explanations

    def to_latex(self) -> str:
        """Format steps as a LaTeX align environment.
           To render it in a Jupyter notebook, use: ``display(Math(latex_string))``."""

        if not self.steps:
            return r"\begin{align}\end{align}"

        lines = []
        for i, (step, explanation) in enumerate(zip(self.steps, self._explanations)):
            if step is None or step == 'None':
                # Note-only transitional row (e.g. the improper-integral limit
                # annotation): render no left-hand expression, just the note.
                line = rf"& \quad \quad"
            else:
                line = r"&=" + cached_latex(step) if i > 0 else "&" + cached_latex(step)
            if explanation:
                line += r" \quad " + format_note(explanation)
            lines.append(line)

        body = r" \\ ".join(lines)
        return rf"\begin{{align}}{body}\end{{align}}"
