from typing import Dict
from sympy import Matrix, latex, sympify

from domains import RefCalculator, SVDSolver

svd_solver = SVDSolver()
ref_calculator = RefCalculator()


class MatrixAnalyzer:
    """A class for analyzing mathematical matrices and computing their properties.

    This analyzer can compute various matrix properties such as rank, determinant,
    eigenvalues, eigenvectors, and reduced row echelon form (REF).
    """

    def _parse_matrix_expression(self, matrix_input: str) -> Matrix:
        """Parse a string representation of a matrix into a SymPy Matrix.

        Args:
            matrix_input (str): String representation of the matrix in format '[[1, 2], [3, 4]]'

        Returns:
            Matrix: A SymPy Matrix object

        Raises:
            ValueError: If the matrix input cannot be parsed
        """
        try:
            return Matrix(sympify(matrix_input))
        except Exception as e:
            raise ValueError("无法解析矩阵输入. 请使用有效的矩阵格式 '[[1, 2], [3, 4]]'") from e

    def analyze(self, matrix_expr: str, analysis_types: str = None) -> Dict:
        """
        Perform comprehensive analysis on a matrix.

        Args:
            matrix_expr (str): String representation of the matrix
            analysis_types (list, optional): List of analysis types to perform.
                                           Defaults to ['ref', 'rank', 'determinant',
                                           'eigenvalues', 'eigenvectors', 'svd']

        Returns:
            dict: Dictionary containing analysis results with keys:
                - dimensions: Matrix dimensions as "n×m"
                - is_square: Boolean indicating if matrix is square
                - ref: Reduced row echelon form (if requested)
                - rank: Matrix rank (if requested)
                - determinant: Matrix determinant (if requested and applicable)
                - is_invertible: Boolean indicating if matrix is invertible (if determinant computed)
                - eigenvalues: List of eigenvalues (if requested and applicable)
                - eigenvectors: List of eigenvectors (if requested and applicable)

        Raises:
            ValueError: If matrix analysis fails
        """
        if analysis_types is None:
            analysis_types = ['ref', 'rank', 'determinant',
                              'eigenvalues', 'eigenvectors', 'svd']
        results = {}

        try:
            matrix = self._parse_matrix_expression(matrix_expr)
            n, m = matrix.shape

            results['dimensions'] = f"{n}×{m}"
            results['is_square'] = n == m

            if 'ref' in analysis_types:
                try:
                    ref_result = ref_calculator.compute_latex(
                        matrix_expr, 'ref')
                    results['ref'] = ref_result
                except Exception as e:
                    results['ref'] = str(e)

            # Calculate rank
            if 'rank' in analysis_types:
                try:
                    results['rank'] = latex(matrix.rank())
                except Exception as e:
                    results['rank'] = str(e)

            if results['is_square']:
                # Calculate determinant (only for square matrices)
                if 'determinant' in analysis_types:
                    try:
                        det_value = matrix.det()
                        results['determinant'] = latex(det_value)
                        results['is_invertible'] = det_value != 0
                    except Exception as e:
                        results['determinant'] = str(e)

                # Calculate eigenvalues and eigenvectors
                if 'eigenvalues' in analysis_types:
                    try:
                        eigenvals = matrix.eigenvals()
                        # Process eigenvalues (may contain repeated roots)
                        eigenvalues_list = []
                        for val, multiplicity in eigenvals.items():
                            for _ in range(multiplicity):
                                eigenvalues_list.append(latex(val))

                        results['eigenvalues'] = eigenvalues_list
                    except Exception as e:
                        results['eigenvalues'] = str(e)

                if 'eigenvectors' in analysis_types:
                    try:
                        eigenvects = matrix.eigenvects()
                        eigenvectors_list = []

                        for val, multiplicity, vectors in eigenvects:
                            for vec in vectors:
                                eigenvectors_list.append(latex(vec))

                        results['eigenvectors'] = eigenvectors_list
                    except Exception as e:
                        results['eigenvectors'] = str(e)

            else:
                results['determinant'] = r"\text{矩阵非方阵, 无法计算行列式}"
                results['eigenvalues'] = r"\text{矩阵非方阵, 无法计算特征值}"
                results['eigenvectors'] = r"\text{矩阵非方阵, 无法计算特征向量}"

            return results

        except Exception as e:
            raise ValueError(f"矩阵分析失败: {str(e)}") from e


if __name__ == "__main__":
    analyzer = MatrixAnalyzer()
    expr = "[[1, 2], [3, 4]]"
    res = analyzer.analyze(expr)
