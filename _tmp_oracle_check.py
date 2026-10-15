import sys, re, json
sys.path.insert(0, r'F:\python_project\HighDream')
import sympy
from sympy import Integral, Derivative, Limit, Sum, Symbol, diff, simplify, trigsimp, N, oo
from utils.cache_utils import cached_sympify
x = Symbol('x')
NL = re.compile(r'\\newline')
SPLIT = re.compile(r'\\quad|\\qquad|\\text')


def lastval(latex):
    body = NL.sub('\n', latex)
    rows = [r for r in body.split('\n') if '&=' in r]
    if not rows:
        return None
    r = rows[-1].split('&=', 1)[1]
    r = SPLIT.split(r)[0]
    return r.strip()


def ntextnotes(latex):
    return len(re.findall(r'\\text\{', latex))


data = json.load(open('/tmp/engine_out/notebook-weierstrass.json', encoding='utf-8'))
for c in data:
    obj = cached_sympify(c['content'])
    cls = type(obj).__name__
    fv = lastval(c['latex'])
    info = {'i': c['i'], 'class': cls,
            'nlines': len([r for r in NL.sub('\n', c['latex']).split('\n') if '&=' in r]),
            'final': (fv[:130] if fv else None)}
    if isinstance(obj, Integral) and len(obj.limits) == 1 and len(obj.limits[0]) == 1:
        integ = obj.args[0]
        try:
            anti = cached_sympify(fv)
            resid = simplify(trigsimp(diff(anti, x) - integ))
            info['kind'] = 'indefinite'
            info['resid'] = sympy.sstr(resid)[:120]
            info['zero'] = (resid == 0)
        except Exception as e:
            info['kind'] = 'indefinite'
            info['note'] = 'parse fail: ' + str(e)[:60]
    elif isinstance(obj, (Derivative, Limit, Sum)) or (isinstance(obj, Integral) and len(obj.limits[0]) >= 2):
        gt = obj.doit()
        info['kind'] = 'doit'
        info['gt'] = sympy.sstr(gt)[:90]
        try:
            f = cached_sympify(fv)
            info['equal'] = bool(simplify(trigsimp(f - gt)) == 0)
            if not info['equal']:
                pts = [0.3, 0.7, 1.3, -0.4, 2.1]
                ok = True
                for p in pts:
                    try:
                        ok = ok and abs(complex(N((f - gt).subs(x, p)))) < 1e-4
                    except Exception:
                        pass
                info['numclose'] = ok
        except Exception as e:
            info['note'] = str(e)[:50]
    else:
        info['kind'] = 'other'
    info['ntext'] = ntextnotes(c['latex'])
    print(json.dumps(info, ensure_ascii=False))
