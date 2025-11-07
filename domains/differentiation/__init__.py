from utils import MatcherList, RuleDict

from .rules.diff_basic_rules import (
    add_matcher, add_rule, chain_matcher, chain_rule,
    div_rule, mul_div_matcher, mul_rule
)
from .rules.diff_function_rules import (
    acos_matcher, acos_rule, asin_matcher, asin_rule,
    atan_matcher, atan_rule, const_matcher, const_rule,
    cos_matcher, cos_rule, cosh_matcher, cosh_rule,
    cot_matcher, cot_rule, csc_matcher, csc_rule,
    exp_matcher, exp_rule, log_matcher, log_rule,
    pow_matcher, pow_rule, sec_matcher, sec_rule,
    sin_matcher, sin_rule, sinh_matcher, sinh_rule,
    tan_matcher, tan_rule, tanh_matcher, tanh_rule,
    var_matcher, var_rule
)

# Rule mapping: Rule name -> Rule function
RULE_DICT: RuleDict = {
    'add': add_rule,
    'mul': mul_rule,
    'div': div_rule,
    'chain': chain_rule,
    'const': const_rule,
    'var': var_rule,
    'pow': pow_rule,
    'sin': sin_rule,
    'cos': cos_rule,
    'tan': tan_rule,
    'sec': sec_rule,
    'csc': csc_rule,
    'cot': cot_rule,
    'asin': asin_rule,
    'acos': acos_rule,
    'atan': atan_rule,
    'exp': exp_rule,
    'log': log_rule,
    'sinh': sinh_rule,
    'cosh': cosh_rule,
    'tanh': tanh_rule
}

# Note: Earlier entries have higher priority.
MATCHER_LIST: MatcherList = [
    const_matcher, var_matcher, add_matcher, mul_div_matcher,
    chain_matcher, pow_matcher, sin_matcher, cos_matcher, tan_matcher,
    sec_matcher, csc_matcher, cot_matcher, asin_matcher, acos_matcher,
    atan_matcher, exp_matcher, log_matcher, sinh_matcher, cosh_matcher,
    tanh_matcher
]
