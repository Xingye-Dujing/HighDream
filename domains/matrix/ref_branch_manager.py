from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from sympy import Eq, Ne, simplify_logic, And, Or, satisfiable
from sympy.core.relational import Relational
from sympy import Expr
from sympy import simplify, latex, Matrix


@dataclass
class BranchNode:
    """表示一个分支节点: 条件集合 + 当前矩阵 + 已记录步骤 + 状态"""
    conditions: List[Relational] = field(default_factory=list)
    matrix: Matrix = None
    steps: List[Tuple[Matrix, str]] = field(
        default_factory=list)
    finished: bool = False
    reason: Optional[str] = None  # 如果分支不可行, 记录原因

    def add_condition(self, cond: Relational):
        self.conditions.append(cond)

    def extend_conditions(self, conds: List[Relational]):
        self.conditions.extend(conds)

    def add_step(self, matrix, explanation: str):
        self.steps.append((matrix, explanation))


class BranchManager:
    """管理分支的生成, 可满足性检测与合并等价叶子"""

    @staticmethod
    def cond_eq(expr: Expr) -> Relational:
        return Eq(expr, 0)

    @staticmethod
    def cond_ne(expr: Expr) -> Relational:
        return Ne(expr, 0)

    @staticmethod
    def is_satisfiable(conditions: List[Relational]) -> bool:
        """用 sympy.satisfiable 判断条件集合是否可满足"""
        if not conditions:
            return True
        combined = And(*conditions)
        sat = satisfiable(combined)
        return bool(sat)

    @staticmethod
    def merge_leaves(leaves: List[BranchNode]) -> List[BranchNode]:
        """
        合并得到相同结果的叶子.
        相同矩阵(按 simplify 后字符串表示), 合并其条件为 OR(...)
        简化合并条件, 若结果为 False 则丢弃该组
        返回合并后的叶子列表(每个元素 finished=True)
        """
        # 对矩阵所有元素做 simplify, 再转字符串
        def canonical_matrix_key(matrix):
            try:
                m = matrix.applyfunc(lambda x: simplify(x) if x != 0 else 0)
            except Exception:
                m = matrix
            return str(m)

        groups = {}
        for leaf in leaves:
            if leaf.reason:
                # 不可行分支不入组
                continue
            key = canonical_matrix_key(leaf.matrix)
            groups.setdefault(key, []).append(leaf)

        merged = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
                continue
            # 合并条件
            cond_sets = [g.conditions for g in group]
            ors = Or(*[And(*conds) if conds else True for conds in cond_sets])
            simplified_ors = simplify_logic(ors, force=True)
            if simplified_ors is False:
                # 合并后不可满足
                continue
            # 以第一个分支为基础, 合并步骤并添加合并说明
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
        if not conds:
            return "无额外条件"
        try:
            return latex(And(*conds))
        except Exception:
            return " \\land ".join(latex(c) for c in conds)
