"""Integral-specific manual step solver."""

from typing import Dict

from core.base_manual_step_solver import BaseManualStepSolver
from domains.integral.integral_calculator import SelectIntegralCalculator


_RULE_DISPLAY_NAMES: Dict[str, str] = {
    'add': '加法展开',
    'const': '常数积分',
    'var': '变量积分',
    'mul_const': '常数乘法',
    'pow': '幂函数积分',
    'exp': '指数函数积分',
    'log': '对数函数积分',
    'sin': '正弦函数积分',
    'cos': '余弦函数积分',
    'tan': '正切函数积分',
    'sec': '正割函数积分',
    'csc': '余割函数积分',
    'cot': '余切函数积分',
    'sinh': '双曲正弦积分',
    'cosh': '双曲余弦积分',
    'tanh': '双曲正切积分',
    'sech': '双曲正割积分',
    'csch': '双曲余割积分',
    'coth': '双曲余切积分',
    'inverse_trig': '反三角函数积分',
    'inverse_tangent_linear': '线性反正切积分',
    'sin_power': '正弦幂积分',
    'cos_power': '余弦幂积分',
    'tan_power': '正切幂积分',
    'logarithmic': '对数函数积分',
    'parts': '分部积分法',
    'substitution': '换元积分法',
    'f_x_mul_exp_g_x': 'f(x)e^g(x) 型积分',
    'quotient_diff_form': '商微分形式',
    'quadratic_sqrt_reciprocal': '二次根式倒数积分',
    'sqrt_div_sqrt': '根式相除积分',
    'weierstrass_substitution': '万能公式代换',
}


class IntegralManualStepSolver(BaseManualStepSolver):
    """Manual step-by-step solver for integration."""

    _domain = 'integral'
    rule_display_names = _RULE_DISPLAY_NAMES

    def _create_calculator(self):
        return SelectIntegralCalculator()
