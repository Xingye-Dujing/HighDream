from typing import List
from utils import RuleDict, MatcherList
from .limit_step_generator import LimitStepGenerator
from .limit_help_func import (
    check_combination_indeterminate,
    check_function_tends_to_zero,
    check_limit_exists,
    check_limit_exists_oo,
    get_limit_args,
    is_constant,
    is_indeterminate_form,
    is_infinite,
    is_zero,
)
from .rules.base_rules import (
    small_o_add_rule, small_o_add_matcher,
    const_inf_mul_rule, const_inf_mul_matcher,
    const_inf_add_rule, const_inf_add_matcher,
    const_inf_div_rule, const_inf_div_matcher,
    const_zero_div_rule, const_zero_div_matcher,
    add_split_rule, add_split_matcher,
    mul_split_rule, mul_split_matcher,
    div_split_rule, div_split_matcher,
    direct_substitution_rule, direct_substitution_matcher,
    conjugate_rationalize_matcher, conjugate_rationalize_rule
)
from .rules.important_rules import (
    ln_one_plus_x_over_x_matcher, ln_one_plus_x_over_x_rule,
    one_plus_one_over_x_pow_x_matcher, one_plus_one_over_x_pow_x_rule,
    sin_over_x_matcher, sin_over_x_rule,
    exp_minus_one_over_x_rule, exp_minus_one_over_x_matcher,
    g_over_sin_rule, g_over_sin_matcher,
    g_over_ln_one_plus_rule, g_over_ln_one_plus_matcher,
    g_over_exp_minus_one_rule, g_over_exp_minus_one_matcher,
)
from .rules.function_rules import (
    acos_matcher, acos_rule, asin_matcher, asin_rule, atan_matcher, atan_rule,
    cos_matcher, cos_rule, cosh_matcher, cosh_rule,
    cot_matcher, cot_rule, csc_matcher, csc_rule, exp_matcher, exp_rule,
    log_matcher, log_rule, pow_matcher, pow_rule, sec_matcher, sec_rule,
    sin_matcher, sin_rule, sinh_matcher, sinh_rule, tan_matcher, tan_rule,
    tanh_matcher, tanh_rule
)
from .rules.lhopital_rules import (
    lhopital_direct_matcher, lhopital_direct_rule,
    lhopital_zero_times_inf_matcher, lhopital_zero_times_inf_rule,
    lhopital_inf_minus_inf_matcher, lhopital_inf_minus_inf_rule,
    lhopital_power_matcher, lhopital_power_rule
)
from .rules.taylor_rules import (
    taylor_quotient_rule, taylor_quotient_matcher,
    taylor_substitution_rule, taylor_substitution_matcher,
    taylor_infinity_rule, taylor_infinity_matcher,
    taylor_composition_rule, taylor_composition_matcher
)

__all__ = [
    'LimitStepGenerator',
    'check_combination_indeterminate',
    'check_function_tends_to_zero',
    'check_limit_exists',
    'check_limit_exists_oo',
    'get_limit_args',
    'is_constant',
    'is_indeterminate_form',
    'is_infinite',
    'is_zero',
    'LimitStepGenerator',
]

# Lhopital rules' names
LHOPITAL_RULES: List[str] = [
    'lhopital_direct_rule',
    'lhopital_zero_times_inf_rule',
    'lhopital_inf_minus_inf_rule',
    'lhopital_power_rule'
]

# Rule mapping: Rule name -> Rule function
RULE_DICT: RuleDict = {
    'small_o_add': small_o_add_rule,
    'const_inf_add': const_inf_add_rule,
    'const_inf_mul': const_inf_mul_rule,
    'const_inf_div': const_inf_div_rule,
    'const_zero_div': const_zero_div_rule,
    'direct_substitution': direct_substitution_rule,
    'add_split': add_split_rule,
    'mul_split': mul_split_rule,
    'div_split': div_split_rule,
    'pow': pow_rule,
    'exp': exp_rule,
    'log': log_rule,
    'sin': sin_rule,
    'cos': cos_rule,
    'tan': tan_rule,
    'sec': sec_rule,
    'csc': csc_rule,
    'cot': cot_rule,
    'asin': asin_rule,
    'acos': acos_rule,
    'atan': atan_rule,
    'sinh': sinh_rule,
    'cosh': cosh_rule,
    'tanh': tanh_rule,
    'sin_over_x': sin_over_x_rule,
    'one_plus_one_over_x_pow_x': one_plus_one_over_x_pow_x_rule,
    'ln_one_plus_x_over_x': ln_one_plus_x_over_x_rule,
    'exp_minus_one_over_x': exp_minus_one_over_x_rule,
    'g_over_sin': g_over_sin_rule,
    'g_over_ln_one_plus': g_over_ln_one_plus_rule,
    'g_over_exp_minus_one': g_over_exp_minus_one_rule,
    'lhopital_direct': lhopital_direct_rule,
    'lhopital_zero_times_inf': lhopital_zero_times_inf_rule,
    'lhopital_inf_minus_inf': lhopital_inf_minus_inf_rule,
    'lhopital_power': lhopital_power_rule,
    'taylor_quotient': taylor_quotient_rule,
    'taylor_substitution': taylor_substitution_rule,
    'taylor_infinity': taylor_infinity_rule,
    'taylor_composition': taylor_composition_rule,
    'conjugate_rationalize': conjugate_rationalize_rule,
}

# Matchers are applied in order — earlier entries have higher priority.
# General strategy:
#   1. Handle special asymptotic forms (e.g., O(...), oo+-c, 0/0, etc.)
#   2. Try direct substitution (when safe)
#   3. Apply algebraic tricks (e.g., conjugate rationalization)
#   4. Match standard limits (sin(x)/x, (1+1/x)^x, etc.)
#   5. Split arithmetic operations
#   6. Transform power/exponential/log forms
#   7. Apply Lhopital rules (only for indeterminate forms)
#   8. Fall back to elementary function expansions (no indeterminacy check!)
MATCHER_LIST: MatcherList = [
    # Asymptotic simplifications (small-o, const +-oo, etc.)
    small_o_add_matcher,
    const_inf_add_matcher,
    const_inf_mul_matcher,
    const_inf_div_matcher,
    const_zero_div_matcher,
    # Safe direct substitution (both sub-limits exist and are determinate)
    direct_substitution_matcher,
    # Algebraic rationalization (e.g., sqrt(a) - sqrt(b))
    conjugate_rationalize_matcher,
    # Standard limits: sin(x)/x, (1+1/x)^x, ln(1+x)/x, (e^x-1)/x
    sin_over_x_matcher,
    one_plus_one_over_x_pow_x_matcher,
    ln_one_plus_x_over_x_matcher,
    exp_minus_one_over_x_matcher,
    # Reciprocal standard limits: x/sin(x), x/ln(1+x), x/(e^x-1)
    g_over_sin_matcher,
    g_over_ln_one_plus_matcher,
    g_over_exp_minus_one_matcher,
    # Arithmetic splitting: (f+g), (f*g), (f/g)
    add_split_matcher,
    mul_split_matcher,
    div_split_matcher,
    # Power/log transformation (general a^b → exp(b*ln a))
    pow_matcher,
    # Lhopital rules — only for indeterminate forms
    lhopital_power_matcher,             # 0^0, 1^oo, oo^0
    lhopital_inf_minus_inf_matcher,     # oo-oo
    lhopital_direct_matcher,            # 0/0, oo/oo, 0*oo → converted to quotient
    # oo*0 (placed after direct quotient to avoid redundancy)
    lhopital_zero_times_inf_matcher,
    # Elementary function matchers (NO indeterminacy check — must come last!)
    exp_matcher, log_matcher,
    sin_matcher, cos_matcher, tan_matcher,
    sec_matcher, csc_matcher, cot_matcher,
    asin_matcher, acos_matcher, atan_matcher,
    sinh_matcher, cosh_matcher, tanh_matcher,
]
