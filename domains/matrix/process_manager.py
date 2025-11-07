# TODO 后端返回 task_id， 前端引入轮询: 实现异步处理, 避免前端卡死

from sympy import latex

from domains import DeterminantCalculator, DetCalculator, Rank, EigenSolver, SVDSolver

det_cal_1 = DeterminantCalculator()
det_cal_2 = DetCalculator()
rank = Rank()
eigen = EigenSolver()
svd = SVDSolver()


class ProcessManager:
    @staticmethod
    def get_process(process_type, expression):
        res = {'sucess': True, 'process': ''}
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
            res['sucess'] = False

        return res
