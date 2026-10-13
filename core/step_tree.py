"""Structured, per-step solver output that powers step-linking in the UI.

`structured_solve` runs the same engine as the ``*_steps`` LaTeX output but
returns machine-readable data: each derivation step carries its LaTeX, the
rule explanation, and any unevaluated sub-expressions (pending Integral /
Derivative / Limit) that the user can open in a fresh, prefilled cell -- the
"每一步都可以链接到新单元格再计算" feature.
"""

from typing import Any, Dict, List

from sympy import Derivative, Integral, Limit, Symbol, latex

from utils.cache_utils import cached_sympify


def _var_name(spec: Any) -> str:
    """Extract the differentiation/integration symbol name from a SymPy spec.

    Handles ``x`` (Symbol), ``(x,)``, ``(x, order)`` and ``(x, lower, upper)``.
    """
    if spec is None:
        return 'x'
    if isinstance(spec, Symbol):
        return str(spec)
    if hasattr(spec, '__iter__'):
        items = list(spec)
        if items and isinstance(items[0], Symbol):
            return str(items[0])
    return str(spec)


def _subtask_of(atom) -> Dict[str, Any]:
    """Reduce an unevaluated operation atom to a prefilled-cell descriptor."""
    if isinstance(atom, Derivative):
        spec = atom.args[1] if len(atom.args) > 1 else None
        return {'op': 'diff', 'expression': str(atom.args[0]),
                'variable': _var_name(spec)}
    if isinstance(atom, Integral):
        spec = {'op': 'integral', 'expression': str(atom.args[0])}
        second = atom.args[1] if len(atom.args) >= 2 else None
        if isinstance(second, Symbol):
            spec['variable'] = str(second)
        elif second is not None and hasattr(second, '__iter__'):
            items = list(second)
            spec['variable'] = _var_name(items)
            if len(items) >= 3:
                spec['lower'] = str(items[1])
                spec['upper'] = str(items[2])
        else:
            spec['variable'] = 'x'
        return spec
    if isinstance(atom, Limit):
        spec = {'op': 'limit', 'expression': str(atom.args[0]),
                'variable': _var_name(atom.args[1]) if len(atom.args) > 1 else 'x'}
        if len(atom.args) > 2:
            spec['point'] = str(atom.args[2])
        if len(atom.args) > 3:
            spec['direction'] = str(atom.args[3])
        return spec
    return {'op': '', 'expression': str(atom), 'variable': 'x'}


def _subtasks_of(expr) -> List[Dict[str, Any]]:
    """Collect deduplicated pending-operation subtasks inside one step."""
    seen = set()
    result: List[Dict[str, Any]] = []
    for atom in expr.atoms(Derivative, Integral, Limit):
        task = _subtask_of(atom)
        signature = (task['op'], task['expression'], task.get('variable'),
                     task.get('point'), task.get('upper'))
        if signature in seen:
            continue
        seen.add(signature)
        result.append(task)
    return result


def _compute_steps(op_type: str, data: Dict[str, Any]):
    """Run the matching calculator and return (exprs, explanations)."""
    expression = data.get('expression', '')
    var = Symbol(data.get('variable', 'x') or 'x')

    if op_type == 'diff':
        # Local import: avoids a domains<->core package cycle at module load.
        # pylint: disable=import-outside-toplevel
        from domains import DiffCalculator
        return DiffCalculator().compute_list(expression, var)

    if op_type == 'integral':
        # pylint: disable=import-outside-toplevel
        from domains import IntegralCalculator
        lower = data.get('lower', '')
        upper = data.get('upper', '')
        bounds = {}
        if str(upper).strip() != '':
            bounds['upper'] = cached_sympify(str(upper))
            bounds['lower'] = cached_sympify(str(lower)) if str(lower).strip() else 0
        return IntegralCalculator().compute_list(expression, var, **bounds)

    if op_type == 'limit':
        # pylint: disable=import-outside-toplevel
        from domains import LimitCalculator
        point = data.get('point', '0')
        direction = data.get('direction', '+')
        return LimitCalculator().compute_list(expression, var, cached_sympify(str(point)), direction)

    raise ValueError(f'不支持的操作类型: {op_type}')


def structured_solve(op_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``{success, op, steps:[{latex, reason, subtasks:[...]}]}``.

    Any computation error yields ``{success: False, error}`` so the endpoint
    stays exception-free.
    """
    try:
        exprs, explanations = _compute_steps(op_type, data)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {'success': False, 'error': str(exc)}

    steps: List[Dict[str, Any]] = []
    for idx, (expr, reason) in enumerate(zip(exprs, explanations)):
        step_latex = '' if expr is None else latex(expr)
        pending = _subtasks_of(expr) if hasattr(expr, 'atoms') else []
        # The final resolved step has no pending operations; still expose it.
        steps.append({
            'index': idx,
            'latex': step_latex,
            'reason': reason,
            'subtasks': pending,
        })

    return {'success': True, 'op': op_type, 'steps': steps}
