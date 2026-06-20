"""Differentiation-specific manual step solver."""

from typing import Dict

from core.base_manual_step_solver import BaseManualStepSolver
from domains.differentiation.diff_calculator import SelectDiffCalculator


_RULE_DISPLAY_NAMES: Dict[str, str] = {
    'add': '加法法则',
    'mul': '乘法法则',
    'div': '除法法则',
    'chain': '链式法则',
    'const': '常数求导',
    'var': '变量求导',
    'pow': '幂函数求导',
    'sin': '正弦求导',
    'cos': '余弦求导',
    'tan': '正切求导',
    'sec': '正割求导',
    'csc': '余割求导',
    'cot': '余切求导',
    'asin': '反正弦求导',
    'acos': '反余弦求导',
    'atan': '反正切求导',
    'exp': '指数求导',
    'log': '对数求导',
    'sinh': '双曲正弦求导',
    'cosh': '双曲余弦求导',
    'tanh': '双曲正切求导',
}


class DiffManualStepSolver(BaseManualStepSolver):
    """Manual step-by-step solver for differentiation."""

    _domain = 'diff'
    rule_display_names = _RULE_DISPLAY_NAMES

    def _create_calculator(self):
        return SelectDiffCalculator()
