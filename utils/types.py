from typing import Dict, List, Optional, Protocol, Tuple, Union
from sympy import Derivative, Determinant, Expr, Integral, Limit, Symbol

# The operation to be performed.
Operation = Union[Derivative, Determinant, Integral, Limit]
# The evaluation context providing additional information.
Context = Union[Symbol, Expr, str]
RuleContext = Dict[str, Context]
RuleFunctionReturn = Optional[Tuple[Expr, str]]
MatcherFunctionReturn = Optional[str]


class RuleFunction(Protocol):
    """Callable that applies a rule to a symbolic expression,

    Returns:
        Tuple:
        - Expr: The transformed expression.
        - str: The description of the rule applied.
    """

    def __call__(self, expr: Expr,
                 context: RuleContext) -> RuleFunctionReturn: ...


class MatcherFunction(Protocol):
    """Callable that matches a symbolic expression

    Returns:
        The name of the matched rule as a string, or None if no rule applies.
    """

    def __call__(self, expr: Expr,
                 context: RuleContext) -> MatcherFunctionReturn: ...


# Mapping of rule names to functions.
RuleDict = Dict[str, RuleFunction]
RuleList = List[RuleFunction]
MatcherList = List[MatcherFunction]
