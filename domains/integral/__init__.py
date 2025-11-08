from utils import MatcherList, RuleDict

from .rules.parts_rule_help import select_parts_u_dv
from .rules.substitution_rule_help import (
    try_exp_log_substitution,
    try_radical_substitution,
    try_standard_substitution,
    try_trig_substitution
)
from .rules.integral_basic_rules import (
    add_matcher, add_rule, mul_const_matcher, mul_const_rule
)
from .rules.integral_function_rules import (
    const_matcher, const_rule, exp_matcher, exp_rule,
    hyperbolic_matcher, hyperbolic_rule, inverse_trig_matcher, inverse_trig_rule,
    log_matcher, log_rule, power_matcher, power_rule,
    trig_matcher, trig_rule, var_matcher, var_rule
)
from .rules.integral_special_rules import (
    parts_matcher, parts_rule, substitution_matcher, substitution_rule
)

__all__ = [
    'RULE_DICT',
    'MATCHER_LIST',
    'select_parts_u_dv',
    'try_exp_log_substitution',
    'try_radical_substitution',
    'try_standard_substitution',
    'try_trig_substitution'
]

# Rule mapping: Rule name -> Rule function
RULE_DICT: RuleDict = {
    'power': power_rule,
    'exp': exp_rule,
    'log': log_rule,
    'trig': trig_rule,
    'inverse_trig': inverse_trig_rule,
    'hyperbolic': hyperbolic_rule,
    'const': const_rule,
    'var': var_rule,
    'add': add_rule,
    'mul_const': mul_const_rule,
    'substitution': substitution_rule,
    'parts': parts_rule,
}

# Note: Earlier entries have higher priority.
MATCHER_LIST: MatcherList = [
    const_matcher, var_matcher, add_matcher, mul_const_matcher,
    power_matcher, exp_matcher, log_matcher, trig_matcher,
    inverse_trig_matcher, hyperbolic_matcher, substitution_matcher,
    parts_matcher
]
