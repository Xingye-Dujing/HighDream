from abc import ABC

from sympy import Matrix, latex, simplify

from core.matrix_step_generator import MatrixStepGenerator
from core.base_step_generator import format_note
from utils.cache_utils import cached_sympify


class CommonMatrixCalculator(ABC):
    """Abstract base class for matrix calculation operations.

    Provides common functionality for matrix calculations including step tracking,
    matrix parsing, and simplification utilities."""

    def __init__(self) -> None:
        """Initialize the calculator with a step generator."""
        self.step_generator = MatrixStepGenerator()

    def add_step(self, title: str) -> None:
        """Add a step title to the calculation process."""
        text = str(title)
        if '$' in text:
            self.step_generator.add_step(format_note(text))
        elif text.lstrip().startswith('\\'):
            self.step_generator.add_step(text)
        else:
            self.step_generator.add_step(f"\\text{{{text}}}")

    @staticmethod
    def is_square(matrix: Matrix) -> bool:
        """True when a matrix has as many rows as columns."""
        return matrix.rows == matrix.cols

    def add_matrix(self, matrix: str | Matrix, name: str = "A") -> None:
        """Add a matrix to the calculation steps display."""
        self.step_generator.add_step(f"{name} = {latex(matrix)}")

    def add_vector(self, vector: Matrix, name: str = "x") -> None:
        """Add a vector to the calculation steps display."""
        self.step_generator.add_step(f"{name} = {latex(vector)}")

    def add_equation(self, equation: str) -> None:
        """Add an equation to the calculation steps display.

        Parameters:
            equation (str): The LaTeX representation of the equation."""

        self.step_generator.add_step(equation)

    def get_steps_latex(self) -> str:
        """Get the complete calculation steps in LaTeX format."""
        return self.step_generator.get_steps_latex()

    @staticmethod
    def parse_matrix_input(matrix_input: str | Matrix) -> Matrix:
        """Parse matrix input string into a SymPy Matrix."""
        try:
            if isinstance(matrix_input, Matrix):
                return matrix_input
            parsed = cached_sympify(matrix_input)
            if isinstance(parsed, dict):
                raise ValueError('矩阵不能是字典/空集合')
            matrix = Matrix(parsed)
            if matrix.rows == 0 or matrix.cols == 0:
                raise ValueError('矩阵不能为空')
            return matrix
        except Exception as e:
            raise ValueError(
                f"Unable to parse matrix input: {matrix_input}, Error: {str(e)}") from e

    def parse_vector_input(self, vector_input: str | Matrix) -> Matrix:
        """Parse vector input string into a SymPy Matrix (column vector)."""
        try:
            if isinstance(vector_input, Matrix):
                return vector_input
            if isinstance(vector_input, dict):
                raise ValueError('向量不能是字典/空集合')
            # Handle column vector format, e.g. '[[1],[2],[3]]'
            if vector_input.startswith('[[') and vector_input.endswith(']]'):
                vector = Matrix(cached_sympify(vector_input))
            # Handle row vector format, e.g. '[1,2,3]'
            else:
                # Convert to column vector format
                vector_str = vector_input.strip('[]')
                elements = [cached_sympify(x.strip())
                            for x in vector_str.split(',')]
                if any(isinstance(item, dict) for item in elements):
                    raise ValueError('向量元素不能是字典/集合')
                vector = Matrix(elements)
            if vector.rows == 0 or vector.cols == 0:
                raise ValueError('向量不能为空')
            return vector
        except Exception as e:
            raise ValueError(
                f"Unable to parse vector input: {vector_input}, Error: {str(e)}") from e

    @staticmethod
    def simplify_matrix(matrix: Matrix) -> Matrix:
        """Simplify each element in a matrix."""
        return matrix.applyfunc(lambda x: simplify(x) if x != 0 else 0)
