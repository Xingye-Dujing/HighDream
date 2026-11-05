from typing import Dict, Union

from sympy import latex

from domains import DetCalculator, DeterminantCalculator, EigenSolver
from .SVD_solver import SVDSolver
from .rank import Rank

class ProcessManager:  # pylint: disable=too-few-public-methods
    """Manager class for handling different matrix computation processes.

    This class routes requests to appropriate calculators based on process type
    and returns formatted results with LaTeX representations where applicable.
    """

    @staticmethod
    def get_process(process_type: str, expression: str) -> Dict[str, Union[bool, str]]:
        """Process a matrix computation request based on the specified type.

        Args:
            process_type (str): Type of computation to perform.
                Supported types: 'det_1', 'det_2', 'rank', 'eigen', 'svd'
            expression (str): Mathematical expression representing the matrix.

        Returns:
            dict: Result dictionary containing:
                - success (bool): Whether the operation was successful.
                - process (str): LaTeX formatted computation steps.
                - U, S, V (str, optional): LaTeX formatted SVD decomposition matrices.
        """
        res = {'success': True, 'process': ''}
        # Fresh calculator per request: these objects hold mutable step state, so a
        # module-level singleton let concurrent requests interleave and leak each
        # other's steps into the returned derivation.
        if process_type == 'det_1':
            res['process'] = DeterminantCalculator().compute_latex(expression)
        elif process_type == 'det_2':
            res['process'] = DetCalculator().compute_latex(expression)
        elif process_type == 'rank':
            calculator = Rank()
            calculator.auto_matrix_rank(expression)
            res['process'] = calculator.get_steps_latex()
        elif process_type == 'eigen':
            calculator = EigenSolver()
            calculator.auto_eigen_solver(expression)
            res['process'] = calculator.get_steps_latex()
        elif process_type == 'svd':
            calculator = SVDSolver()
            U, S, V = calculator.compute_svd(expression)
            res['process'] = calculator.get_steps_latex()
            res['U'], res['S'], res['V'] = latex(U), latex(S), latex(V)
        else:
            res['success'] = False

        return res
