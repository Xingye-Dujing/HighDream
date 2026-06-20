"""Limit-specific manual step solver."""

from typing import Any, Dict

from sympy import Expr

from core.base_manual_step_solver import BaseManualStepSolver
from domains.limit.limit_calculator import SelectLimitCalculator


_RULE_DISPLAY_NAMES: Dict[str, str] = {
    'sin_over_x': '重要极限 sin(x)/x',
    'one_plus_one_over_x_pow_x': '重要极限 (1+1/x)^x',
    'ln_one_plus_x_over_x': '重要极限 ln(1+x)/x',
    'exp_minus_one_over_x': '重要极限 (e^x-1)/x',
    'g_over_sin': 'g(x)/sin(x) 型',
    'g_over_ln_one_plus': 'g(x)/ln(1+x) 型',
    'g_over_exp_minus_one': 'g(x)/(e^x-1) 型',
    'mul_split': '乘法拆分',
    'add_split': '加法拆分',
    'div_split': '除法拆分',
    'direct_substitution': '直接代入',
    'conjugate_rationalize': '共轭有理化',
    'small_o_add': '小 o 加法',
    'const_inf_add': '常数+无穷',
    'const_inf_mul': '常数×无穷',
    'const_inf_div': '常数/无穷',
    'const_zero_div': '常数/零',
    'lhopital_direct': '洛必达法则 (0/0, ∞/∞)',
    'lhopital_zero_times_inf': '洛必达 (0·∞)',
    'lhopital_inf_minus_inf': '洛必达 (∞-∞)',
    'lhopital_power': '洛必达 (幂指型)',
    'pow': '幂指转换',
    'exp': '指数函数极限',
    'log': '对数函数极限',
    'sin': '正弦函数极限',
    'cos': '余弦函数极限',
    'tan': '正切函数极限',
    'sec': '正割函数极限',
    'csc': '余割函数极限',
    'cot': '余切函数极限',
    'asin': '反正弦极限',
    'acos': '反余弦极限',
    'atan': '反正切记限',
    'sinh': '双曲正弦极限',
    'cosh': '双曲余弦极限',
    'tanh': '双曲正切极限',
}


class LimitManualStepSolver(BaseManualStepSolver):
    """Manual step-by-step solver for limits.

    Extends the base class with:
      * injecting ``point`` and ``direction`` into the per-expression
        calculator context;
      * snapshotting/restoring the l'Hôpital usage counter so that
        merely *listing* applicable rules does not count against the cap.
    """

    _domain = 'limit'
    rule_display_names = _RULE_DISPLAY_NAMES

    def _create_calculator(self):
        return SelectLimitCalculator()

    def _init_calculator(self) -> None:
        # Limit uses a specialized step generator + rationalization setting;
        # reset the wrapped calculator's l'Hôpital counter for this session.
        self.calculator._lhopital_count = 0

    def _extend_context(self, ctx: Dict[str, Any], _expr: Expr) -> None:
        ctx['point'] = self.point
        ctx['direction'] = self.direction

    def _snapshot_solver_state(self) -> Dict[str, Any]:
        return {
            'lhopital_count': getattr(self.calculator, '_lhopital_count', None),
        }

    def _restore_solver_state(self, snapshot: Dict[str, Any]) -> None:
        count = snapshot.get('lhopital_count')
        if count is not None:
            self.calculator._lhopital_count = count
