from typing import Dict, List, Tuple, Callable, Set, Optional

from sympy import (
    Expr, preorder_traversal, srepr, sympify, expand, factor,
    expand_trig, logcombine, expand_log, apart, cancel, radsimp, tan, sin, cos,
    powsimp, trigsimp
)


# from sympy import latex
# from matplotlib import pyplot as plt
# from matplotlib import rcParams

# rcParams["font.sans-serif"] = ["SimHei"]  # Chinese font
# rcParams["axes.unicode_minus"] = False    # Support negative sign


class ExpressionParser:
    """A class for parsing and transforming mathematical expressions using various symbolic computation rules.

    This parser applies transformation rules to generate equivalent forms of an expression up to a specified depth.
    It can also visualize the derivation tree of transformations applied to an expression.
    """

    def __init__(self, max_depth: int = 3, sort_strategy: Optional[Callable[[Expr], float]] = None) -> None:
        """Initialize the ExpressionParser with maximum derivation depth and optional sorting strategy.

        Args:
            max_depth: Maximum depth of the derivation tree (default: 3)
            sort_strategy: Optional function to sort derived expressions by a custom criterion
        """
        self.max_depth = max_depth
        self.sort_strategy = sort_strategy
        self.transform_rules: List[Tuple[str, Callable[[Expr], List[Expr]]]] = []
        self._initialize_rules()

    @staticmethod
    def _try_transform(expr: Expr, func: Callable, *args, **kwargs) -> List[Expr]:
        """Safely apply a transformation function to an expression.

        Args:
            expr: The expression to transform
            func: The transformation function to apply
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function.

        Returns:
            List containing the transformed expression if successful, otherwise empty list.
        """
        try:
            result = func(expr, *args, **kwargs)
            if result is not None and result != expr:
                return [result]
        except Exception:
            return []
        return []

    def _initialize_rules(self) -> None:
        """Initialize the list of transformation rules with their names and corresponding functions."""

        self.transform_rules.extend([
            ("三角展开", self._expand_trig_transform),
            ("积化和差", self._product_or_trig_expand_transform),
            ("多项式展开", self._polynomial_expand_transform),
            ("因式分解", self._factor_transform),
            ("tan 拆开", self._tan_to_sin_cos_transform),
            ("对数合并", self._combine_logarithms_transform),
            ("对数展开", self._expand_logarithmic_transform),
            ("部分分式", self._apart_transform),
            ("分式约分", self._cancel_transform),
            ("有理化", self._radsimp_transform),
            ("幂化简", self._powsimp_transform),
            ("三角化简", self._reciprocal_trig_transform),
        ])

    def parse(self, expression: str) -> List[Tuple[Expr, str]]:
        """Parse an expression and generate all possible derivations within the maximum depth.

        Args:
            expression: String representation of the mathematical expression

        Returns:
            List of tuples containing derived expressions, and their derivation steps.
        """
        expr = sympify(expression)

        def canonical_key(e: Basic | Expr) -> str:
            return srepr(e)

        seen: Set[str] = {canonical_key(expr)}
        derivations: List[Tuple[Expr, str]] = []
        queue = [(expr, 0, "原始表达式")]

        while queue:
            current_expr, depth, reason = queue.pop(0)
            if depth >= self.max_depth:
                continue

            for rule_name, rule in self.transform_rules:
                for new_expr in self._apply_rule_to_subexpressions(rule, current_expr):
                    key = canonical_key(new_expr)
                    if key not in seen:
                        seen.add(key)
                        step_reason = f"{reason} -> {rule_name}"
                        derivations.append((new_expr, step_reason))
                        queue.append((new_expr, depth + 1, step_reason))

        if self.sort_strategy:
            derivations.sort(key=lambda x: self.sort_strategy(x[0]))

        return derivations

    def parse_tree(self, expression: str) -> Dict:
        """Generate a derivation tree for an expression.

        Args:
            expression: String representation of the mathematical expression

        Returns:
            Dictionary representing the derivation tree structure
        """
        expr = sympify(expression)

        def canonical_key(e: Basic | Expr) -> str:
            return srepr(e)

        seen: Set[str] = {canonical_key(expr)}
        node_id = 0

        def make_node(e, reason):
            nonlocal node_id
            node = {"id": node_id, "expr": e,
                    "reason": reason, "children": []}
            node_id += 1
            return node

        tree = make_node(expr, "原始表达式")
        queue = [(expr, 0, tree)]

        while queue:
            current_expr, depth, parent_node = queue.pop(0)
            if depth >= self.max_depth:
                continue

            for rule_name, rule in self.transform_rules:
                for new_expr in self._apply_rule_to_subexpressions(rule, current_expr):
                    key = canonical_key(new_expr)
                    if key not in seen:
                        seen.add(key)
                        child_node = make_node(new_expr, rule_name)
                        parent_node["children"].append(child_node)
                        queue.append((new_expr, depth + 1, child_node))

        return tree

    # def draw_expr_tree(self, tree: Dict, save_path: Optional[str] = None) -> None:
    #     """
    #     Visualize the derivation tree using matplotlib.

    #     Args:
    #         tree: The derivation tree structure to visualize
    #         save_path: Optional path to save the visualization as an image file
    #     """
    #     def _count_nodes(node):
    #         return 1 + sum(_count_nodes(child) for child in node.get("children", []))

    #     def _tree_depth(node):
    #         if not node.get("children"):
    #             return 1
    #         return 1 + max(_tree_depth(child) for child in node.get("children", []))

    #     num_nodes = _count_nodes(tree)
    #     depth = _tree_depth(tree)

    #     # Dynamically adjust canvas size
    #     width = num_nodes * 1.5 # Width adjusted by number of nodes
    #     height = depth * 1.5 # Height adjusted by tree depth

    #     _, ax = plt.subplots(figsize=(width, height))
    #     ax.axis('off')

    #     positions = {}
    #     edges = []
    #     x_counter = [0]  # Used for horizontal layout

    #     def assign_positions(node, depth_level=0):
    #         if not node["children"]:
    #             positions[node["id"]] = (x_counter[0], -depth_level)
    #             x_counter[0] += 1
    #         else:
    #             child_x_positions = []
    #             for child in node["children"]:
    #                 edges.append((node["id"], child["id"]))
    #                 assign_positions(child, depth_level + 1)
    #                 child_x_positions.append(positions[child["id"]][0])
    #             positions[node["id"]] = (
    #                 sum(child_x_positions) / len(child_x_positions),
    #                 -depth_level
    #             )

    #     assign_positions(tree)

    #     # Draw edges
    #     for parent_id, child_id in edges:
    #         x0, y0 = positions[parent_id]
    #         x1, y1 = positions[child_id]
    #         ax.plot([x0, x1], [y0, y1], color="black")

    #     # Draw nodes
    #     for node_id, (x, y) in positions.items():
    #         node = self._find_node_by_id(tree, node_id)
    #         label = f"${latex(node['expr'])}$\n({node['reason']})"
    #         ax.text(x, y, label, ha="center", va="center",
    #                 bbox={"boxstyle": "round,pad=0.5", "fc": "lightblue", "ec": "black"})

    #     if save_path:
    #         plt.savefig(save_path, bbox_inches="tight") # Save to file
    #         print(f"可视化树已保存到：{save_path}")

    def _find_node_by_id(self, node: Dict, node_id: int) -> Dict | None:
        """Find a node in the tree by its ID.

        Args:
            node: The root node to start searching from
            node_id: The ID of the node to find.

        Returns:
            The node with the matching ID or None if not found.
        """
        if node["id"] == node_id:
            return node
        for child in node.get("children", []):
            res = self._find_node_by_id(child, node_id)
            if res:
                return res
        return None

    @staticmethod
    def _apply_rule_to_subexpressions(rule: Callable[[Expr], List[Expr]], expr: Expr) -> List[Expr]:
        """Apply a transformation rule to all subexpressions of an expression.

        Args:
            rule: The transformation rule function to apply
            expr: The expression to which the rule will be applied.

        Returns:
            List of all valid transformed expressions
        """
        results = []
        added_keys = set()

        def add_candidate(e: Expr):
            key = srepr(e)
            if key not in added_keys and e != expr:
                added_keys.add(key)
                results.append(e)

        try:
            for t in rule(expr):
                add_candidate(t)
        except Exception:
            pass

        try:
            for sub in preorder_traversal(expr):
                if sub == expr:
                    continue
                try:
                    for t in rule(sub):  # type:ignore
                        try:
                            new_expr = expr.xreplace({sub: t})
                        except Exception:
                            new_expr = expr.subs(sub, t)
                        add_candidate(new_expr)
                except Exception:
                    pass
        except Exception:
            pass

        return results

    # Transformation rules
    def _polynomial_expand_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand)

    def _factor_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, factor)

    def _product_or_trig_expand_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand_trig)

    def _expand_trig_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand_trig)

    def _combine_logarithms_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, logcombine, force=True)

    def _expand_logarithmic_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, expand_log, force=True)

    def _apart_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, apart)

    def _cancel_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, cancel)

    def _radsimp_transform(
            self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, radsimp)

    def _powsimp_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, powsimp, force=True)

    def _reciprocal_trig_transform(self, expr: Expr) -> List[Expr]:
        return self._try_transform(expr, trigsimp, reciprocal=True)

    @staticmethod
    def _tan_to_sin_cos_transform(expr: Expr) -> List[Expr]:
        """Transform tangent functions into sine over cosine ratios.

        Args:
            expr: The expression potentially containing tangent functions

        Returns:
            List containing the transformed expression or empty list if no transformation occurred.
        """
        try:
            replaced = expr.replace(lambda e: getattr(e, "func", None) == tan,
                                    lambda e: sin(e.args[0]) / cos(e.args[0]))
            if replaced != expr:
                return [replaced]
        except Exception:
            pass
        return []
