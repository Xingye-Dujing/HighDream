from typing import List, Union
from sympy import Expr, Matrix, latex, simplify

from core import BaseStepGenerator


class RefStepGenerator(BaseStepGenerator):
    """Generator for Row Echelon Form (REF) step-by-step solutions."""

    def get_latex(self, steps: List[Union[Expr, Matrix]], explanations: List[str]) -> str:
        """Generate LaTeX formatted output suitable for matrices (supports branch titles, etc.)

        Args:
            steps (list): List of computational steps, can contain Matrix objects or strings
            explanations (list): Corresponding list of explanatory text for each step

        Returns:
            str: LaTeX formatted string with aligned equations and explanations
        """
        latex_str = "\\begin{align}\n"
        for i, (step, explanation) in enumerate(zip(steps, explanations)):
            # Step can be a string (branch title or note) or a Matrix
            if isinstance(step, str):
                # Render as a full line note
                step_str = f"& \\text{{{step}}} \\quad & \\text{{{explanation}}}"
            else:
                # Process matrix or general expression
                try:
                    if isinstance(step, Matrix):
                        m_latex = self._matrix_to_latex(step)
                    else:
                        m_latex = latex(step)
                except Exception:
                    m_latex = str(step)
                # If explanation contains branching keywords, display matrix with explanation directly
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

    def _matrix_to_latex(self, matrix: Matrix) -> str:
        """Convert a matrix to its LaTeX representation with simplified elements.

        Args:
            matrix (Matrix): The matrix to convert

        Returns:
            str: LaTeX representation of the simplified matrix
        """
        simplified_matrix = matrix.applyfunc(
            lambda x: simplify(x) if x != 0 else 0)
        return latex(simplified_matrix)
