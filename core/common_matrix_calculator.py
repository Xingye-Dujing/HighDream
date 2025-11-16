from abc import ABC
from sympy import Matrix, latex, simplify, sympify

from core import MatrixStepGenerator


class CommonMatrixCalculator(ABC):
    """Abstract base class for matrix calculation operations.

    Provides common functionality for matrix calculations including step tracking,
    matrix parsing, and simplification utilities.
    """

    def __init__(self) -> None:
        """Initialize the calculator with a step generator."""
        self.step_generator = MatrixStepGenerator()

    def add_step(self, title: str) -> None:
        """Add a step title to the calculation process."""
        self.step_generator.add_step(f"\\text{{{title}}}")

    def add_matrix(self, matrix: Matrix, name: str = "A") -> None:
        """Add a matrix to the calculation steps display."""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_vector(self, vector: Matrix, name: str = "x") -> None:
        """Add a vector to the calculation steps display."""
        self.step_generator.add_step(f"{name} = {latex(vector)}")

    def add_equation(self, equation: str) -> None:
        """Add an equation to the calculation steps display.

        Parameters:
            equation (str): The LaTeX representation of the equation.
        """
        self.step_generator.add_step(equation)

    def get_steps_latex(self) -> str:
        """Get the complete calculation steps in LaTeX format."""
        return self.step_generator.get_steps_latex()

    def parse_matrix_input(self, matrix_input: str) -> Matrix:
        """Parse matrix input string into a SymPy Matrix."""
        try:
            return Matrix(sympify(matrix_input))
        except Exception as e:
            raise ValueError(
                f"Unable to parse matrix input: {matrix_input}, Error: {str(e)}") from e

    def parse_vector_input(self, vector_input: str) -> Matrix:
        """Parse vector input string into a SymPy Matrix (column vector)."""
        try:
            # Handle column vector format, e.g. '[[1],[2],[3]]'
            if vector_input.startswith('[[') and vector_input.endswith(']]'):
                vector = Matrix(sympify(vector_input))
            # Handle row vector format, e.g. '[1,2,3]'
            else:
                # Convert to column vector format
                vector_str = vector_input.strip('[]')
                elements = [sympify(x.strip())
                            for x in vector_str.split(',')]
                vector = Matrix(elements)
            return vector
        except Exception as e:
            raise ValueError(
                f"Unable to parse vector input: {vector_input}, Error: {str(e)}") from e

    def simplify_matrix(self, matrix: Matrix) -> Matrix:
        """Simplify each element in a matrix."""
        return matrix.applyfunc(lambda x: simplify(x) if x != 0 else 0)
