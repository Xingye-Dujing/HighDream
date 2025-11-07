from sympy import Expr

from utils import Context, MatcherFunction, MatcherList, RuleDict, RuleFunction, RuleList


class RuleRegistry:
    """Manages the registration and matching of transformation rules for symbolic expressions."""

    def __init__(self) -> None:
        self._rules: RuleDict = {}
        self._matchers: MatcherList = []

    def _register_rule(self, name: str, func: RuleFunction) -> None:
        """Register a rule by name."""
        self._rules[name] = func

    def _register_matcher(self, matcher: MatcherFunction) -> None:
        """Register a matcher."""
        self._matchers.append(matcher)

    def _register_all_rules(self, rules: RuleDict) -> None:
        """Register all rules in a dictionary."""
        for name, func in rules.items():
            self._register_rule(name, func)

    def _register_all_matchers(self, matchers: MatcherList) -> None:
        """Register all matchers in a list."""
        for matcher in matchers:
            self._register_matcher(matcher)

    def register_all(self, rules: RuleDict, matchers: MatcherList) -> None:
        """Register all rules and matchers."""
        self._register_all_rules(rules)
        self._register_all_matchers(matchers)

    def get_applicable_rules(self, expr: Expr, context: Context) -> RuleList:
        """Return all rules applicable to the given expression."""
        applicable = []
        for matcher in self._matchers:
            rule_name = matcher(expr, context)
            if rule_name in self._rules:
                applicable.append(self._rules[rule_name])
        return applicable
