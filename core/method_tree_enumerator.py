"""Exhaustive rule-enumeration tree builder.

Given a problem (domain + expression + parameters), the
:class:`MethodTreeEnumerator` explores every rule application path the
manual step solver can take and records the result as a tree. Hard
cutoffs (depth, total node count, wall-clock time, external cancel)
keep the search bounded so a single request cannot pin a worker thread.

The enumerator is designed to be driven by
:class:`routes.task_manager.TaskManager` in a background thread while
the HTTP layer polls :meth:`snapshot` for incremental progress updates.
"""

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

from sympy import latex

from config import (
    METHOD_TREE_DEFAULT_MAX_DEPTH,
    METHOD_TREE_DEFAULT_MAX_NODES,
    METHOD_TREE_DEFAULT_TIME_SECONDS,
    METHOD_TREE_HARD_MAX_DEPTH,
    METHOD_TREE_HARD_MAX_NODES,
    METHOD_TREE_HARD_MAX_TIME_SECONDS,
)
from core.manual_step_solver import ManualStepSolver


def _clamp(value: Any, default: int, hard_max: int, name: str) -> int:
    """Coerce ``value`` to int and clamp to ``[1, hard_max]``."""
    try:
        num = int(value)
    except (TypeError, ValueError):
        num = default
    if num <= 0:
        num = default
    num = min(num, hard_max)
    if num < 1:
        raise ValueError(f"Cutoff '{name}' must be >= 1 (got {num}).")
    return num


class MethodTreeEnumerator:
    """BFS enumerator that builds a tree of all rule-application paths."""

    def __init__(  # pylint: disable=too-many-positional-arguments
        self,
        domain: str,
        expression: str,
        variable: str = 'x',
        point: Any = 0,
        direction: str = '+',
        max_depth: Any = METHOD_TREE_DEFAULT_MAX_DEPTH,
        max_nodes: Any = METHOD_TREE_DEFAULT_MAX_NODES,
        time_limit_seconds: Any = METHOD_TREE_DEFAULT_TIME_SECONDS,
        interactive: bool = False,
    ) -> None:
        self.domain = domain
        self.expression = expression
        self.variable = variable
        self.point = point
        self.direction = direction
        self.interactive = interactive

        self.max_depth = _clamp(
            max_depth, METHOD_TREE_DEFAULT_MAX_DEPTH,
            METHOD_TREE_HARD_MAX_DEPTH, 'max_depth')
        self.max_nodes = _clamp(
            max_nodes, METHOD_TREE_DEFAULT_MAX_NODES,
            METHOD_TREE_HARD_MAX_NODES, 'max_nodes')
        # Interactive mode has no time limit.
        if interactive:
            self.time_limit = float('inf')
        else:
            self.time_limit = _clamp(
                time_limit_seconds, METHOD_TREE_DEFAULT_TIME_SECONDS,
                METHOD_TREE_HARD_MAX_TIME_SECONDS, 'time_limit_seconds')

        # Incrementally-built tree.
        self._nodes: Dict[str, Dict[str, Any]] = {}
        self._children: Dict[str, List[str]] = {}
        self._node_counter = 0
        self._root_id: Optional[str] = None

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._start_time = 0.0

        # Interactive mode: pause-and-resume protocol.
        self._decision_condition = threading.Condition()
        self._pending_decision: Optional[Dict[str, Any]] = None
        self._decision_response: Optional[bool] = None

        # Outcome fields (set by run()).
        self.truncated: bool = False
        self.reason: str = 'completed'
        self.max_depth_seen: int = 0
        self.error: Optional[str] = None

    # ------------------------------------------------------------------ public

    def run(self) -> Dict[str, Any]:
        """Execute BFS and return the complete tree payload."""
        self._start_time = time.monotonic()
        try:
            self._build_tree()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.error = f"{type(e).__name__}: {e}"
            self.truncated = True
            if self.reason == 'completed':
                self.reason = 'error'
        return self._finalize_payload()

    def cancel(self) -> None:
        """Request cooperative cancellation. run() returns at the next safe point."""
        self._stop.set()
        with self._decision_condition:
            self._decision_response = False
            self._decision_condition.notify()

    def respond_decision(self, accepted: bool) -> bool:
        """Respond to a pending interactive decision. Returns True if a decision was pending."""
        with self._decision_condition:
            if self._pending_decision is None:
                return False
            self._decision_response = accepted
            self._decision_condition.notify()
            return True

    def snapshot(self) -> Dict[str, Any]:
        """Return a consistent view of the tree built so far (thread-safe)."""
        with self._lock:
            nodes_copy = {k: dict(v) for k, v in self._nodes.items()}
            children_copy = {k: list(v) for k, v in self._children.items()}
            pd = self._pending_decision
            pending = dict(pd) if pd is not None else None
            payload = {
                'root_id': self._root_id,
                'nodes': nodes_copy,
                'children': children_copy,
                'pending_decision': pending,
            }
        payload['stats'] = self._build_stats(len(nodes_copy))
        return payload

    # ----------------------------------------------------------------- internal

    def _elapsed(self) -> float:
        return time.monotonic() - self._start_time if self._start_time else 0.0

    def _should_stop(self) -> bool:
        """Check every cutoff condition (non-destructive). Returns True to stop."""
        if len(self._nodes) >= self.max_nodes:
            self.truncated, self.reason = True, 'node_limit'
            return True
        if not self.interactive and self._elapsed() >= self.time_limit:
            self.truncated, self.reason = True, 'time_limit'
            return True
        if self._stop.is_set():
            self.truncated, self.reason = True, 'cancelled'
            return True
        return False

    def _new_node_id(self) -> str:
        nid = f"n{self._node_counter}"
        self._node_counter += 1
        return nid

    def _make_root_solver(self):
        """Construct a fresh solver for the configured problem."""
        return ManualStepSolver(
            domain=self.domain,
            expression=self.expression,
            variable=self.variable,
            point=self.point,
            direction=self.direction,
        )

    def _replay_solver(self, rule_path: List[str]):
        """Build a fresh solver and replay ``rule_path`` from the root state."""
        solver = self._make_root_solver()
        for rule_name in rule_path:
            if solver.done:
                break
            try:
                state = solver.apply_rule(rule_name)
            except Exception:  # pylint: disable=broad-exception-caught
                break
            if state.get('error'):
                break
        return solver

    @staticmethod
    def _current_latex(solver) -> str:
        """Best-effort LaTeX for the solver's current working expression."""
        expr = solver.current_expr
        if expr is not None:
            try:
                return latex(expr)
            except Exception:  # pylint: disable=broad-exception-caught
                return repr(expr)
        if solver.steps:
            last = solver.steps[-1]
            try:
                return latex(last)
            except Exception:  # pylint: disable=broad-exception-caught
                return ''
        return ''

    def _register_node(  # pylint: disable=too-many-positional-arguments
        self,
        parent_id: Optional[str],
        depth: int,
        latex_text: str,
        top_latex_text: str,
        rule_applied: Optional[str],
        rule_display: Optional[str],
        explanation: str,
        done: bool,
        truncated: bool,
        final_latex: Optional[str],
    ) -> str:
        """Allocate a node ID, record the node, and link it to its parent."""
        nid = self._new_node_id()
        self._nodes[nid] = {
            'id': nid,
            'parent': parent_id,
            'depth': depth,
            'latex': latex_text,
            'top_latex': top_latex_text,
            'rule_applied': rule_applied,
            'rule_display': rule_display,
            'explanation': explanation,
            'done': done,
            'children': [],
            'truncated': truncated,
            'final_latex': final_latex,
        }
        self._children[nid] = []
        if parent_id is not None and parent_id in self._children:
            self._children[parent_id].append(nid)
            self._nodes[parent_id]['children'] = list(self._children[parent_id])
        self.max_depth_seen = max(self.max_depth_seen, depth)
        return nid

    def _build_tree(self) -> None:
        """Core BFS loop."""
        try:
            root_solver = self._make_root_solver()
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.error = f"构建初始求解器失败: {e}"
            self.truncated = True
            self.reason = 'error'
            return

        with self._lock:
            self._root_id = self._register_node(
                parent_id=None,
                depth=0,
                latex_text=self._current_latex(root_solver),
                top_latex_text=self._current_latex(root_solver),
                rule_applied=None,
                rule_display=None,
                explanation='',
                done=root_solver.done,
                truncated=False,
                final_latex=None,
            )

        # BFS work queue: (node_id, rule_path_from_root, depth)
        queue: Deque[Tuple[str, List[str], int]] = deque()
        if not root_solver.done:
            queue.append((self._root_id, [], 0))

        while queue:
            if self._should_stop():
                break

            parent_id, rule_path, depth = queue.popleft()

            if depth >= self.max_depth:
                with self._lock:
                    node = self._nodes.get(parent_id)
                    if node is not None:
                        node['truncated'] = True
                self.truncated = True
                if self.reason == 'completed':
                    self.reason = 'depth_limit'
                continue

            solver = self._replay_solver(rule_path)
            if solver.done:
                with self._lock:
                    node = self._nodes.get(parent_id)
                    if node is not None and node['final_latex'] is None:
                        node['done'] = True
                        node['final_latex'] = self._current_latex(solver)
                        # If back-substitution happened, show pre-back-subs
                        # in this node and create a child for the back-subs.
                        _subs = solver.calculator.step_generator.subs_dict
                        if _subs:
                            _steps = solver.calculator.step_generator.steps
                            if len(_steps) >= 2:
                                try:
                                    _pre_latex = latex(_steps[-2])
                                    _post_latex = latex(_steps[-1])
                                    node['final_latex'] = _pre_latex
                                    self._register_node(
                                        parent_id=parent_id,
                                        depth=node.get('depth', 0) + 1,
                                        latex_text=_post_latex,
                                        top_latex_text=_post_latex,
                                        rule_applied='back_subs',
                                        rule_display='回代换元变量',
                                        explanation='',
                                        done=True,
                                        truncated=False,
                                        final_latex=_post_latex,
                                    )
                                except Exception:  # pylint: disable=broad-exception-caught
                                    pass
                continue

            try:
                rules = solver.applicable_rules()
            except Exception:  # pylint: disable=broad-exception-caught
                rules = []

            if not rules:
                with self._lock:
                    node = self._nodes.get(parent_id)
                    if node is not None:
                        node['done'] = True
                        node['final_latex'] = self._current_latex(solver)
                continue

            for rule in rules:
                if self._should_stop():
                    break

                child_rule_path = rule_path + [rule['name']]
                child_solver = self._replay_solver(child_rule_path)

                # If replay failed (error set), drop this branch silently.
                if child_solver.error:
                    continue

                child_done = child_solver.done
                child_latex = self._current_latex(child_solver)
                final = child_latex if child_done else None
                # Check if back-substitution happened (e.g., u-substitution,
                # trig substitution, radical substitution).
                # When it did, steps[-1] is the back-substituted result and
                # steps[-2] (or _current_top_operation) is the expression
                # BEFORE back-substitution. We create a dedicated child node
                # for the back-substitution step.
                _subs_dict = child_solver.calculator.step_generator.subs_dict
                _back_subs_latex = None
                if child_done and _subs_dict:
                    try:
                        _pre = latex(child_solver._current_top_operation())
                        _back_subs_latex = child_latex  # steps[-1] = post-back-subs
                        child_latex = _pre
                        final = _pre
                    except Exception:  # pylint: disable=broad-exception-caught
                        pass
                try:
                    child_top_latex = latex(child_solver._current_top_operation())
                except Exception:  # pylint: disable=broad-exception-caught
                    child_top_latex = child_latex

                if self.interactive:
                    accepted = self._ask_user_decision(
                        parent_id=parent_id,
                        parent_latex=self._current_latex(solver),
                        rule_name=rule['name'],
                        rule_display=rule.get('display_name', rule['name']),
                        preview_latex=child_top_latex,
                    )
                    if not accepted:
                        continue
                    if self._should_stop():
                        break

                with self._lock:
                    child_id = self._register_node(
                        parent_id=parent_id,
                        depth=depth + 1,
                        latex_text=child_latex,
                        top_latex_text=child_top_latex,
                        rule_applied=rule['name'],
                        rule_display=rule.get('display_name', rule['name']),
                        explanation='',
                        done=child_done,
                        truncated=False,
                        final_latex=final,
                    )
                    # If back-substitution was detected, register a dedicated
                    # child node that shows the result in the original variable.
                    if _back_subs_latex is not None:
                        self._register_node(
                            parent_id=child_id,
                            depth=depth + 2,
                            latex_text=_back_subs_latex,
                            top_latex_text=_back_subs_latex,
                            rule_applied='back_subs',
                            rule_display='回代换元变量',
                            explanation='',
                            done=True,
                            truncated=False,
                            final_latex=_back_subs_latex,
                        )

                if not child_done:
                    queue.append((child_id, child_rule_path, depth + 1))

            # After all rules for this node were rejected (interactive mode),
            # mark the node as done so it appears as a leaf.
            if self.interactive and not self._should_stop():
                with self._lock:
                    node = self._nodes.get(parent_id)
                    if node is not None and not node['children']:
                        node['done'] = True
                        node['final_latex'] = self._current_latex(solver)

    def _build_stats(self, node_count: int) -> Dict[str, Any]:
        return {
            'node_count': node_count,
            'max_depth_seen': self.max_depth_seen,
            'elapsed_seconds': round(self._elapsed(), 3),
            'truncated': self.truncated,
            'reason': self.reason,
            'error': self.error,
        }

    def _finalize_payload(self) -> Dict[str, Any]:
        """Produce the final tree payload returned by run()."""
        snap = self.snapshot()
        snap['stats'] = self._build_stats(len(snap['nodes']))
        return snap

    def _ask_user_decision(
        self,
        parent_id: str,
        parent_latex: str,
        rule_name: str,
        rule_display: str,
        preview_latex: str,
    ) -> bool:
        """Pause BFS and wait for user approval/rejection of a rule.

        Sets ``_pending_decision`` (visible via :meth:`snapshot`), then blocks
        on the condition variable until :meth:`respond_decision` is called or
        the enumerator is cancelled.  Returns ``True`` if the user accepted.
        """
        decision = {
            'parent_id': parent_id,
            'parent_latex': parent_latex,
            'rule_name': rule_name,
            'rule_display': rule_display,
            'preview_latex': preview_latex,
        }
        with self._decision_condition:
            if self._should_stop():
                return False
            self._pending_decision = decision
            self._decision_response = None
            # Wait until respond_decision or cancel sets _decision_response.
            while self._decision_response is None:
                if self._stop.is_set():
                    self._pending_decision = None
                    return False
                self._decision_condition.wait(timeout=0.5)
            accepted = self._decision_response
            self._pending_decision = None
            self._decision_response = None
        return accepted
