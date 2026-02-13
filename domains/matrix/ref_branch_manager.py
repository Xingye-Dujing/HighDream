from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from sympy import And, Eq, Expr, Matrix, Ne, Or, latex, simplify, simplify_logic, satisfiable
from sympy.core.relational import Relational


@dataclass
class BranchNode:
    """Represents a branch node: condition set + current matrix + recorded steps + status.

    Attributes:
        conditions: List of relational conditions that define this branch.
        matrix: Current state of the matrix in this branch.
        steps: List of transformation steps (matrix, explanation) taken so far.
        finished: Whether this branch has been fully processed.
        reason: If a branch is infeasible, record the reason.
    """
    conditions: List[Relational] = field(default_factory=list)
    matrix: Matrix = None
    steps: List[Tuple[Matrix, str]] = field(
        default_factory=list)
    finished: bool = False
    reason: Optional[str] = None  # If a branch is infeasible, record the reason

    def add_condition(self, cond: Relational) -> None:
        """Add a single condition to the branch.

        Args:
            cond: A relational condition to add.
        """
        self.conditions.append(cond)

    def extend_conditions(self, conds: List[Relational]) -> None:
        """Extend the conditions' list with multiple conditions.

        Args:
            conds: List of relational conditions to add.
        """
        self.conditions.extend(conds)

    def add_step(self, matrix: str | Matrix, explanation: str) -> None:
        """Record a transformation step in the branch.

        Args:
            matrix: The matrix after transformation.
            explanation: Description of the transformation step.
        """
        self.steps.append((matrix, explanation))


class BranchManager:
    """Manages branch generation, satisfiability checking and merging equivalent leaves."""

    @staticmethod
    def cond_eq(expr: Expr) -> Relational:
        """Create an equality condition (expression == 0).

        Args:
            expr: Expression to check for equality to zero.

        Returns:
            Equality relation.
        """
        return Eq(expr, 0)

    @staticmethod
    def cond_ne(expr: Expr) -> Relational:
        """Create a non-equality condition (expression != 0).

        Args:
            expr: Expression to check for non-equality to zero.

        Returns:
            Non-equality relation.
        """
        return Ne(expr, 0)

    @staticmethod
    def is_satisfiable(conditions: List[Relational]) -> bool:
        """Check if a set of conditions is satisfiable using sympy.satisfiable.

        Args:
            conditions: List of relational conditions.

        Returns:
            True if conditions are satisfiable, False otherwise.
        """
        if not conditions:
            return True
        combined = And(*conditions)
        sat = satisfiable(combined)
        return bool(sat)

    @staticmethod
    def merge_leaves(leaves: List[BranchNode]) -> List[BranchNode]:
        """Merge leaves that result in the same matrix.

        Leaves with the same canonical matrix representation have their conditions
        combined with OR logic. Simplified merged conditions are checked,
        and groups resulting in False are discarded.

        Args:
            leaves: List of branch nodes to potentially merge.

        Returns:
            List of merged branch nodes with finished=True.
        """

        # Canonicalize matrix by simplifying all elements and converting to string
        def canonical_matrix_key(matrix):
            try:
                m = matrix.applyfunc(lambda x: simplify(x) if x != 0 else 0)
            except Exception:
                m = matrix
            return str(m)

        groups = {}
        for leaf in leaves:
            if leaf.reason:
                # Skip infeasible branches
                continue
            key = canonical_matrix_key(leaf.matrix)
            groups.setdefault(key, []).append(leaf)

        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            # Merge conditions
            cond_sets = [g.conditions for g in group]
            ors = Or(*[And(*conds) if conds else True for conds in cond_sets])
            simplified_ors = simplify_logic(ors, force=True)
            if simplified_ors is False:
                # Discard unsatisfiable group
                continue
            # Use the first branch as base, merge steps, and add merge explanation.
            base = group[0]
            merged_node = BranchNode(
                conditions=[simplified_ors] if not isinstance(
                    simplified_ors, bool) else ([] if simplified_ors else [Eq(1, 0)]),
                matrix=base.matrix,
                steps=base.steps.copy(),
                finished=True
            )
            merged_node.add_step(
                f"等价结果来自 {len(group)} 个分支的合并", "合并说明")
            merged.append(merged_node)

        return merged

    @staticmethod
    def conditions_to_latex(conds: List[Relational]) -> str:
        """Convert a list of conditions to LaTeX format.

        Args:
            conds: List of relational conditions.

        Returns:
            LaTeX representation of the conditions.
        """
        if not conds:
            return "无额外条件"
        try:
            return latex(And(*conds))
        except Exception:
            return " \\land ".join(latex(c) for c in conds)
