from .det_step_generator import DetStepGenerator
from .ref_branch_manager import BranchManager, BranchNode
from .ref_step_generator import RefStepGenerator

from .rules.det_advanced_rules import (
    circulant_matcher, circulant_rule,
    symmetric_matcher, symmetric_rule,
    vandermonde_matcher, vandermonde_rule,
)
from .rules.det_basic_rules import (
    diagonal_matcher, diagonal_rule,
    duplicate_column_matcher, duplicate_column_rule,
    duplicate_row_matcher, duplicate_row_rule,
    linear_combination_matcher, linear_combination_rule,
    scalar_multiple_column_matcher, scalar_multiple_column_rule,
    scalar_multiple_row_matcher, scalar_multiple_row_rule,
    triangular_matcher, triangular_rule,
    zero_column_matcher, zero_column_rule,
    zero_row_matcher, zero_row_rule,
)
from .rules.det_expansion_rules import (
    laplace_expansion_matcher, laplace_expansion_rule,
)
from .rules.ref_basic_rules import (
    apply_elimination_rule, apply_scale_rule, apply_swap_rule
)

RULE_DICT = {
    'zero_row': zero_row_rule,
    'zero_column': zero_column_rule,
    'duplicate_row': duplicate_row_rule,
    'duplicate_column': duplicate_column_rule,
    'diagonal': diagonal_rule,
    'triangular': triangular_rule,
    'scalar_multiple_row': scalar_multiple_row_rule,
    'scalar_multiple_column': scalar_multiple_column_rule,
    'linear_combination': linear_combination_rule,
    'laplace_expansion': laplace_expansion_rule,
    'vandermonde': vandermonde_rule,
    'circulant': circulant_rule,
    'symmetric': symmetric_rule,
}

MATCHER_LIST = [
    zero_row_matcher,
    zero_column_matcher,
    duplicate_row_matcher,
    duplicate_column_matcher,
    scalar_multiple_row_matcher,
    scalar_multiple_column_matcher,
    diagonal_matcher,
    triangular_matcher,
    vandermonde_matcher,
    circulant_matcher,
    symmetric_matcher,
    linear_combination_matcher,
    laplace_expansion_matcher,
]
