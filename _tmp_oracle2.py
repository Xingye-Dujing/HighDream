import sys, json
sys.path.insert(0, r'F:\python_project\HighDream')
from sympy import Integral, Derivative, Symbol, N, oo
from utils.cache_utils import cached_sympify
from domains import IntegralCalculator

x = Symbol('x')
data = json.load(open('/tmp/engine_out/notebook-weierstrass.json', encoding='utf-8'))

PAIRS = [(0.2, 0.6), (0.3, 0.9), (0.15, 0.5), (0.25, 0.7), (0.4, 0.8)]


def ftc_check(anti, integ):
    agree = 0
    tried = 0
    details = []
    for a, b in PAIRS:
        try:
            dfa = complex(N(anti.subs(x, b) - anti.subs(x, a)))
            num = complex(N(Integral(integ, (x, a, b)).evalf()))
            tried += 1
            diff_ = abs(dfa - num)
            scale = max(1.0, abs(num))
            rel = diff_ / scale
            ok = rel < 1e-5
            agree += 1 if ok else 0
            details.append('%.2f-%.2f rel=%.2e%s' % (a, b, rel, '' if ok else ' X'))
        except Exception as exc:
            details.append('%s: %s' % (type(exc).__name__, str(exc)[:40]))
            continue
    return agree, tried, details


for c in data:
    i = c['i']
    obj = cached_sympify(c['content'])
    cls = type(obj).__name__
    out = {'i': i, 'class': cls}
    if isinstance(obj, Integral) and len(obj.limits) == 1 and len(obj.limits[0]) == 1:
        integ = obj.args[0]
        steps, _notes = IntegralCalculator().compute_list(integ, x)
        anti = steps[-1]
        out['nsteps'] = len(steps)
        if anti is None or anti.has(Integral):
            out['result'] = 'UNRESOLVED'
        else:
            agree, tried, det = ftc_check(anti, integ)
            out['result'] = 'OK' if (tried > 0 and agree == tried) else ('WRONG' if tried > 0 else 'UNKNOWN')
            out['ftc'] = '%d/%d' % (agree, tried)
            out['det'] = det[:3]
            if agree != tried:
                out['anti'] = str(anti)[:160]
    elif isinstance(obj, Derivative):
        out['kind'] = 'derivative'
    print('JSON ' + json.dumps(out, ensure_ascii=False), flush=True)
print('DONE', flush=True)
