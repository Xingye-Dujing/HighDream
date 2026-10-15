"""Generate engine output + oracle grade for a notebook file.
Engine output = the rule engine's actual step-by-step LaTeX (via start_compute('auto')).
Oracle grade = for INDEFINITE integrals, differentiate the ENGINE'S OWN antiderivative
(_compute_steps('integral', ...) last step) and compare to the integrand (SymPy is the
oracle here, never the producer). Writes /tmp/engine_out/<FILE> and /tmp/grade_<FILE>.
"""
import json
import os
import re
import sys

from sympy import Integral, Derivative, simplify, Symbol

from routes.compute_service import (
    start_compute, _normalize_latex_ops, _split_top_args, _paren_close_index,
)
from core.step_tree import _compute_steps
from utils.cache_utils import cached_sympify
from utils.matrix_latex import latex_to_nested


def find_call(name, text):
    """Return the balanced-paren inner arg string of the first ``name(`` call."""
    m = re.search(r'(?<![A-Za-z_\\])' + re.escape(name) + r'\s*\(', text)
    if not m:
        return None
    op = m.end() - 1
    cl = _paren_close_index(text, op)
    if cl is None:
        return None
    return text[op + 1:cl]


def integrand_and_var(content):
    """Mirror the auto path: normalize, take the int(...) args -> (integrand_str, var)."""
    norm = _normalize_latex_ops(latex_to_nested(content))
    inner = find_call('int', norm)
    if inner is None:
        inner = find_call('integrate', norm)
    if inner is None:
        # fall back to a sympified Integral
        obj = cached_sympify(norm)
        if isinstance(obj, Integral):
            b = obj.args[1] if len(obj.args) > 1 else None
            return str(obj.args[0]), str(b[0]) if b else 'x'
        return None, None
    args = _split_top_args(inner)
    if not args:
        return None, None
    body = args[0]
    var = args[1].strip() if len(args) > 1 and args[1].strip() else 'x'
    return body, var


def grade_indefinite(body, var):
    exprs, _expl = _compute_steps('integral', {'expression': body, 'variable': var})
    concrete = [e for e in exprs if e is not None]
    nsteps = len(exprs)
    F = concrete[-1] if concrete else None
    f = cached_sympify(body)
    if F is None:
        return 'UNRESOLVED', '', nsteps
    if hasattr(F, 'atoms') and F.atoms(Integral, Derivative):
        return 'UNRESOLVED', '', nsteps
    d = F.diff(Symbol(var))
    resid = simplify(d - f)
    return ('OK' if resid == 0 else 'WRONG'), str(resid), nsteps


def main(fname):
    with open(os.path.join('docs', fname), encoding='utf-8') as fh:
        cells = json.load(fh)
    eng_dir = os.path.join(os.environ.get('TMP', '/tmp'), 'engine_out')
    os.makedirs(eng_dir, exist_ok=True)
    engine, grade = [], []
    for i, cell in enumerate(cells):
        content = cell.get('content', '')
        var = cell.get('variable', 'x')
        data = {'operation_type': 'auto', 'expression': content, 'variable': var}
        try:
            ok, res = start_compute('auto', data)
            latex_str = res if isinstance(res, str) else json.dumps(str(res), ensure_ascii=False)
        except Exception as exc:  # noqa
            ok, latex_str = False, f'ENGINE EXCEPTION: {exc}'
        engine.append({'i': i, 'content': content, 'ok': bool(ok), 'latex': latex_str})

        g = {'i': i, 'content': content, 'class': 'Integral',
             'result': None, 'resid': None, 'gt': None, 'nsteps': None}
        try:
            body, ivar = integrand_and_var(content)
            if body is None:
                g['result'] = 'UNRESOLVED'
                g['resid'] = 'parse-failed'
            else:
                ivar = ivar or var
                result, resid, nsteps = grade_indefinite(body, ivar)
                g.update({'result': result, 'resid': resid, 'nsteps': nsteps})
        except Exception as exc:  # noqa
            g['result'] = 'UNRESOLVED'
            g['resid'] = f'grade-exc: {exc}'
        grade.append(g)
        print(f'[{i}] {g["result"]} nsteps={g["nsteps"]} ok={bool(ok)}  {content[:60]}', flush=True)

    base = fname  # keep .json in name to mirror /tmp/grade_<FILE>.json
    with open(os.path.join(eng_dir, base), 'w', encoding='utf-8') as fh:
        json.dump(engine, fh, ensure_ascii=False, indent=1)
    tmp = os.environ.get('TMP', '/tmp')
    with open(os.path.join(tmp, f'grade_{base}'), 'w', encoding='utf-8') as fh:
        json.dump(grade, fh, ensure_ascii=False, indent=1)
    print('WROTE', os.path.join(eng_dir, base), 'and', os.path.join(tmp, f'grade_{base}'))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'notebook-zhihu.json')
