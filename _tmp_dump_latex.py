import sys, json, re
sys.path.insert(0, r'F:\python_project\HighDream')
data = json.load(open('/tmp/engine_out/notebook-weierstrass.json', encoding='utf-8'))
NL = re.compile(r'\\newline')
for c in data:
    body = NL.sub('\n', c['latex'])
    out = ['===== i=%d ok=%s content=%s =====' % (c['i'], c['ok'], c['content'])]
    out.append(body)
    print('\n'.join(out))
