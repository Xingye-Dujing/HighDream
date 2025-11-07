from typing import List, Tuple, Callable, Set, Optional
from sympy import (
    Expr, preorder_traversal, srepr, sympify, expand, factor,
    expand_trig, logcombine, expand_log, apart, cancel, radsimp, tan, sin, cos,
    powsimp, trigsimp, latex
)
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["SimHei"]  # 中文字体
rcParams["axes.unicode_minus"] = False    # 解决负号 '-' 显示问题


class ExpressionParser:
    def __init__(self, max_depth: int = 3, sort_strategy: Optional[Callable[[Expr], float]] = None):
        self.max_depth = max_depth
        self.sort_strategy = sort_strategy
        self.transform_rules: List[Tuple[str,
                                         Callable[[Expr], List[Expr]]]] = []
        self._initialize_rules()

    @staticmethod
    def _try_transform(expr: Expr, func: Callable, *args, **kwargs) -> List[Expr]:
        try:
            result = func(expr, *args, **kwargs)
            if result is not None and result != expr:
                return [result]
        except Exception:
            return []
        return []

    def _initialize_rules(self):
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
            ("倒数化简", self._reciprocal_trig_transform),
        ])

    def parse(self, expression: str) -> List[Tuple[Expr, str]]:
        expr = sympify(expression)

        def canonical_key(expr: Expr) -> str:
            return srepr(expr)

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

    def parse_tree(self, expression: str):
        expr = sympify(expression)

        def canonical_key(expr: Expr) -> str:
            return srepr(expr)

        seen: Set[str] = {canonical_key(expr)}
        node_id = 0

        def make_node(expr, reason):
            nonlocal node_id
            node = {"id": node_id, "expr": expr,
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

    def draw_expr_tree(self, tree, save_path: Optional[str] = None):
        def _count_nodes(node):
            return 1 + sum(_count_nodes(child) for child in node.get("children", []))

        def _tree_depth(node):
            if not node.get("children"):
                return 1
            return 1 + max(_tree_depth(child) for child in node.get("children", []))

        num_nodes = _count_nodes(tree)
        depth = _tree_depth(tree)

        # 动态调整画布大小
        width = num_nodes * 1.5    # 宽度按节点数调整
        height = depth * 1.5       # 高度按树深度调整

        _fig, ax = plt.subplots(figsize=(width, height))
        ax.axis('off')

        positions = {}
        edges = []
        x_counter = [0]  # 用于水平布局

        def assign_positions(node, depth_level=0):
            if not node["children"]:
                positions[node["id"]] = (x_counter[0], -depth_level)
                x_counter[0] += 1
            else:
                child_x_positions = []
                for child in node["children"]:
                    edges.append((node["id"], child["id"]))
                    assign_positions(child, depth_level + 1)
                    child_x_positions.append(positions[child["id"]][0])
                positions[node["id"]] = (
                    sum(child_x_positions) / len(child_x_positions),
                    -depth_level
                )

        assign_positions(tree)

        # 绘制边
        for parent_id, child_id in edges:
            x0, y0 = positions[parent_id]
            x1, y1 = positions[child_id]
            ax.plot([x0, x1], [y0, y1], color="black")

        # 绘制节点
        for node_id, (x, y) in positions.items():
            node = self._find_node_by_id(tree, node_id)
            label = f"${latex(node['expr'])}$\n({node['reason']})"
            ax.text(x, y, label, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.5", fc="lightblue", ec="black"))

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")  # 保存到文件
            print(f"可视化树已保存到：{save_path}")

        plt.show()

    def _find_node_by_id(self, node, node_id):
        if node["id"] == node_id:
            return node
        for child in node.get("children", []):
            res = self._find_node_by_id(child, node_id)
            if res:
                return res
        return None

    def _apply_rule_to_subexpressions(self, rule: Callable[[Expr], List[Expr]], expr: Expr) -> List[Expr]:
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
                    for t in rule(sub):
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

    # 变换规则
    def _polynomial_expand_transform(
        self, expr): return self._try_transform(expr, expand)

    def _factor_transform(self, expr):
        return self._try_transform(expr, factor)

    def _product_or_trig_expand_transform(
        self, expr): return self._try_transform(expr, expand_trig)

    def _expand_trig_transform(
        self, expr): return self._try_transform(expr, expand_trig)

    def _combine_logarithms_transform(
        self, expr): return self._try_transform(expr, logcombine, force=True)

    def _expand_logarithmic_transform(
        self, expr): return self._try_transform(expr, expand_log, force=True)

    def _apart_transform(self, expr):
        return self._try_transform(expr, apart)

    def _cancel_transform(self, expr):
        return self._try_transform(expr, cancel)

    def _radsimp_transform(
        self, expr): return self._try_transform(expr, radsimp)

    def _powsimp_transform(self, expr):
        return self._try_transform(expr, powsimp, force=True)

    def _reciprocal_trig_transform(self, expr):
        return self._try_transform(expr, trigsimp, reciprocal=True)

    def _tan_to_sin_cos_transform(self, expr):
        try:
            replaced = expr.replace(lambda e: getattr(e, "func", None) == tan,
                                    lambda e: sin(e.args[0]) / cos(e.args[0]))
            if replaced != expr:
                return [replaced]
        except Exception:
            pass
        return []
