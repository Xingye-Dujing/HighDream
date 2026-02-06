from utils import MatcherList, RuleDict

from .rules.parts_rule_help import select_parts_u_dv
from .rules.substitution_rule_help import (
    try_radical_substitution,
    try_standard_substitution,
    try_trig_substitution,
    try_undetermined_coeffs_for_radicals
)
from .rules.f_x_mul_exp_g_x_rule_help import handle_fx_mul_exp_gx, special_add_split_exp_term
from .rules.integral_basic_rules import (
    add_matcher, add_rule, mul_const_matcher, mul_const_rule
)
from .rules.integral_function_rules import (
    const_matcher, const_rule, exp_matcher, exp_rule,
    inverse_trig_matcher, inverse_trig_rule,
    log_matcher, log_rule, pow_matcher, pow_rule,
    var_matcher, var_rule, sin_matcher, sin_rule,
    cos_matcher, cos_rule, tan_matcher, tan_rule,
    sec_matcher, sec_rule, csc_matcher, csc_rule,
    cot_matcher, cot_rule, sinh_matcher, sinh_rule,
    cosh_matcher, cosh_rule, tanh_matcher, tanh_rule,
    csch_matcher, sech_matcher,
    coth_matcher, csch_rule, sech_rule, coth_rule,
    inverse_tangent_linear_matcher,
    inverse_tangent_linear_rule,
    sin_power_matcher, sin_power_rule,
    cos_power_matcher, cos_power_rule,
    tan_power_matcher, tan_power_rule,
)
from .rules.integral_special_rules import (
    logarithmic_matcher, logarithmic_rule,
    parts_matcher, parts_rule, substitution_matcher, substitution_rule,
    f_x_mul_exp_g_x_matcher, f_x_mul_exp_g_x_rule,
    weierstrass_substitution_matcher, weierstrass_substitution_rule,
    quotient_diff_form_matcher, quotient_diff_form_rule,
    quadratic_sqrt_reciprocal_matcher, quadratic_sqrt_reciprocal_rule
)

__all__ = [
    'RULE_DICT',
    'MATCHER_LIST',
    'select_parts_u_dv',
    'try_radical_substitution',
    'try_standard_substitution',
    'try_trig_substitution',
    'handle_fx_mul_exp_gx',
    'special_add_split_exp_term'
]

# Rule mapping: Rule name -> Rule function
RULE_DICT: RuleDict = {
    'pow': pow_rule,
    'exp': exp_rule,
    'log': log_rule,
    'sin': sin_rule,
    'cos': cos_rule,
    'tan': tan_rule,
    'sec': sec_rule,
    'csc': csc_rule,
    'cot': cot_rule,
    'sinh': sinh_rule,
    'cosh': cosh_rule,
    'tanh': tanh_rule,
    'sech': sech_rule,
    'csch': csch_rule,
    'coth': coth_rule,
    'inverse_trig': inverse_trig_rule,
    'inverse_tangent_linear': inverse_tangent_linear_rule,
    'sin_power': sin_power_rule,
    'cos_power': cos_power_rule,
    'tan_power': tan_power_rule,
    'const': const_rule,
    'var': var_rule,
    'mul_const': mul_const_rule,
    'add': add_rule,
    'logarithmic': logarithmic_rule,
    'f_x_mul_exp_g_x': f_x_mul_exp_g_x_rule,
    'quotient_diff_form': quotient_diff_form_rule,
    'substitution': substitution_rule,
    'weierstrass_substitution': weierstrass_substitution_rule,
    "quadratic_sqrt_reciprocal": quadratic_sqrt_reciprocal_rule,
    'parts': parts_rule,
}

# Note: Earlier entries have higher priority.
MATCHER_LIST: MatcherList = [
    const_matcher, var_matcher, logarithmic_matcher, quadratic_sqrt_reciprocal_matcher,
    substitution_matcher, f_x_mul_exp_g_x_matcher, quotient_diff_form_matcher,
    inverse_tangent_linear_matcher, add_matcher, mul_const_matcher, pow_matcher, exp_matcher,
    log_matcher, sin_matcher, cos_matcher, tan_matcher, sin_power_matcher, cos_power_matcher,
    tan_power_matcher, sec_matcher, csc_matcher, cot_matcher, exp_matcher, log_matcher,
    sinh_matcher,  cosh_matcher, tanh_matcher, csch_matcher, sech_matcher, coth_matcher,
    inverse_trig_matcher, weierstrass_substitution_matcher, parts_matcher
]
