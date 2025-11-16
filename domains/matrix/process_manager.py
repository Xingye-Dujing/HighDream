# TODO: Backend should return task_id for frontend polling implementation:
#        Achieve asynchronous processing to prevent frontend blocking

from typing import Dict, Union
from sympy import latex

from domains import DetCalculator, DeterminantCalculator, EigenSolver, Rank, SVDSolver
det_cal_1 = DeterminantCalculator()
det_cal_2 = DetCalculator()
rank = Rank()
eigen = EigenSolver()
svd = SVDSolver()


class ProcessManager:
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
            expression (str): Mathematical expression representing the matrix

        Returns:
            dict: Result dictionary containing:
                - success (bool): Whether the operation was successful
                - process (str): LaTeX formatted computation steps
                - U, S, V (str, optional): LaTeX formatted SVD decomposition matrices
        """
        res = {'success': True, 'process': ''}
        if process_type == 'det_1':
            res['process'] = det_cal_1.compute_latex(expression)
        elif process_type == 'det_2':
            res['process'] = det_cal_2.compute_latex(expression)
        elif process_type == 'rank':
            rank.auto_matrix_rank(expression)
            res['process'] = rank.get_steps_latex()
        elif process_type == 'eigen':
            eigen.auto_eigen_solver(expression)
            res['process'] = eigen.get_steps_latex()
        elif process_type == 'svd':
            U, S, V = svd.compute_svd(expression)
            res['process'] = svd.get_steps_latex()
            res['U'], res['S'], res['V'] = latex(U), latex(S), latex(V)
        else:
            res['success'] = False

        return res
