"""Keyboard / template contract tests.

Every visual key must be either a known token (a partial expression the user
combines) or a filled template that the backend accepts.  A new key that is not
classified here fails the suite instead of shipping untested.
"""

import pathlib
import re
from html import unescape

from app import app
from routes.compute_service import start_compute
from routes.op_form import toggle_form
from tests.test_operation_contract import SAMPLES

ROOT = pathlib.Path(__file__).resolve().parent.parent
INDEX = 'templates/index.html'
PANEL = 'templates/matrix_panel.html'
LAB = 'templates/matrix_lab.html'


def parse_keys(path):
    with open(path, encoding='utf-8') as handle:
        text = handle.read()
    keys = []
    for match in re.finditer(r'<button class="keyboard-key"([^>]*)>(.*?)</button>', text, re.S):
        attrs = match.group(1)
        label = re.sub(r'<[^>]+>', '', match.group(2)).strip()
        data = {}
        for name, value in re.findall(r'data-([a-z-]+)="([^"]*)"', attrs):
            data[name] = unescape(value)
        keys.append({'label': label, 'data': data, 'attrs': attrs})
    return keys


def fill_placeholders(template, values):
    out = template
    for value in values:
        out = out.replace(r'\placeholder{}', str(value), 1)
    assert r'\placeholder' not in out, out
    return out


# ---------------------------------------------------------------------------
# Notebook palette
# ---------------------------------------------------------------------------

NOTEBOOK_TOKENS = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '−', '×', '÷', 'xⁿ', 'xᵢ', '(', ')', '[ ]',
    'x', 'y', 'z', 'n', 't', 'a', 'b', 'c', 'π', 'θ', '∞',
}

NOTEBOOK_FILL = {
    '分数': r'\frac{x}{2}',
    '根号': r'\sqrt{x}',
    'n次根': r'\sqrt[3]{x}',
    '导数': r'\frac{d}{dx}(x^{2})',
    '偏导': r'\partial_{x}(x^{2} y)',
    '矩阵': r'\begin{bmatrix}1&2\\3&4\end{bmatrix}',
    '定积分@structure': r'\int_{0}^{1}(x^{2})\, dx',
    '积分': r'\int (x^{2})\, dx',
    '求和': r'\sum_{n=1}^{3}(n)',
    '求积': r'\prod_{n=1}^{3}(n)',
    '极限@structure': r'\lim_{x \to 0}(x)',
    'sin': r'\sin(x)',
    'cos': r'\cos(x)',
    'tan': r'\tan(x)',
    'ln': r'\ln(x)',
    'log': r'\log_{2}(x)',
    'exp': r'\exp(x)',
    'eˣ': r'e^{x}',
    '等价表达式': r'simplify(x+x)',
    '符号矩阵化简': r'msimplify(A*B*Inverse(B))',
    '求导': r'diff(x^{2},x)',
    '不定积分': r'int(x^{2},x)',
    '定积分@operator': r'int(x^{2},x,0,1)',
    '极限@operator': r'limit(x,x,0)',
    '反常积分': r'improper(x^{-2},x,1,\infty)',
    '泰勒展开': r'taylor(exp(x),x,0,3)',
    '泰勒余项': r'taylorerror(exp(x),x,0,3)',
    '切线方程': r'tangent(x^{2},x,1)',
    '线性化': r'linearization(x^{2},x,1)',
    '临界点/极值': r'critical(x^{2},x)',
    '牛顿法': r'newton(x^{2}-2,x,1,5)',
    '中值定理': r'meanvalue(x^{2},x,0,1)',
    '弧长': r'arclength(x,x,0,1)',
    '体积(圆盘)': r'volume(x,x,0,1)',
    '体积(柱壳)': r'shell(x,x,0,1)',
    '曲面面积': r'surface(x,x,0,1)',
    '偏导数': r'diff(x^{2} y,x)',
    '二重积分': r'iterated(x*y,x,y,0,1,0,1)',
    '隐函数求导': r'implicit(x^{2}+y^{2}-1,x)',
    '参数方程求导': r'parametric(\cos(t),\sin(t))',
    '极坐标面积': r'polararea(1+\cos(\theta),\theta,0,\pi)',
    '极坐标斜率': r'polarslope(1+\cos(\theta),\theta)',
    '级数敛散性': r'sum(1/n^{2},n)',
    '收敛半径': r'radius(1/n,n)',
    '幂级数(系数cₙ)': r'powerseries(1/n,n,0)',
    '微分方程求解': r"ode(f'(x)+f(x)-exp(x),x)",
    '数值积分': r'numerical(\sin(x),x,0,\pi,4)',
    '复数化简': r'complex((1+2i)/(3-i))',
    '极坐标形式': r'complexpolar(1+i)',
    '复数求根': r'complexroots(z^{3}-1,z)',
    '棣莫弗高次幂': r'demoivre(1+i,3)',
    '行列式': r'det(\begin{bmatrix}1&2\\3&4\end{bmatrix})',
    '秩': r'rank([[1,2],[3,4]])',
    '求逆': r'inverse([[1,2],[3,4]])',
    '行最简': r'rref([[1,2],[3,4]])',
    '特征值': r'eigen([[1,2],[3,4]])',
    'SVD': r'svd([[1,2],[3,4]])',
    'Schur': r'schur([[1,2],[3,4]])',
    'LU': r'lu([[1,2],[3,4]])',
    '对角化': r'diag([[1,2],[3,4]])',
    '正交化': r'orthogonal([[1,2],[3,4]])',
    '相关性': r'dependence([[1,2],[3,4]])',
    '向量投影': r'project([[1,2]],[[1,0],[0,1]])',
    '矩阵加法': r'add([[1,2],[3,4]],[[5,6],[7,8]])',
    '矩阵减法': r'subtract([[1,2],[3,4]],[[5,6],[7,8]])',
    '矩阵乘法': r'multiply([[1,2],[3,4]],[[5,6],[7,8]])',
}

NOTEBOOK_OPS = {'linear-solver', 'linear-system', 'transform-1', 'transform-2'}


def notebook_sample(label, latex):
    if label == '定积分':
        key = '定积分@structure' if latex.startswith(r'\int_') else '定积分@operator'
    elif label == '极限':
        key = '极限@structure' if latex.startswith(r'\lim') else '极限@operator'
    else:
        key = label
    return NOTEBOOK_FILL.get(key)


def test_every_notebook_key_is_classified():
    keys = parse_keys(INDEX)
    assert len(keys) == 100
    for key in keys:
        data = key['data']
        has_latex = 'latex' in data
        has_op = 'op' in data
        assert has_latex != has_op, key
        if has_op:
            assert data['op'] in NOTEBOOK_OPS, key
        elif data['latex'] in NOTEBOOK_TOKENS or key['label'] in NOTEBOOK_TOKENS:
            continue
        else:
            assert notebook_sample(key['label'], data['latex']) is not None, key


def test_every_notebook_computable_key_works_through_auto():
    keys = parse_keys(INDEX)
    for key in keys:
        data = key['data']
        if 'latex' not in data:
            continue
        if data['latex'] in NOTEBOOK_TOKENS or key['label'] in NOTEBOOK_TOKENS:
            continue
        sample = notebook_sample(key['label'], data['latex'])
        assert sample, key
        ok, result = start_compute('auto', {'operation_type': 'auto', 'expression': sample})
        assert ok, (key['label'], sample, result)
        text = result if isinstance(result, str) else str(result)
        assert '$' not in text
        assert r'\placeholder' not in text
        assert r'\square' not in text


NOTEBOOK_VALUES = {
    '分数': ['x', '2'],
    '根号': ['x'],
    'n次根': ['3', 'x'],
    '导数': ['x', 'x^2'],
    '偏导': ['x', 'x^2 y'],
    '矩阵': ['1', '2', '3', '4'],
    '定积分@structure': ['0', '1', 'x^2', 'x'],
    '积分': ['x^2', 'x'],
    '求和': ['n=1', '3', 'n'],
    '求积': ['n=1', '3', 'n'],
    '极限@structure': ['x', '0', 'x'],
    'sin': ['x'], 'cos': ['x'], 'tan': ['x'], 'ln': ['x'],
    'log': ['2', 'x'], 'exp': ['x'], 'eˣ': ['x'],
    '等价表达式': ['x+x'],
    '符号矩阵化简': ['A*B*Inverse(B)'],
    '求导': ['x^2', 'x'],
    '不定积分': ['x^2', 'x'],
    '定积分@operator': ['x^2', 'x', '0', '1'],
    '极限@operator': ['x', 'x', '0'],
    '反常积分': ['x^-2', 'x', '1', 'oo'],
    '泰勒展开': ['exp(x)', 'x', '0', '3'],
    '泰勒余项': ['exp(x)', 'x', '0', '3'],
    '切线方程': ['x^2', 'x', '1'],
    '线性化': ['x^2', 'x', '1'],
    '临界点/极值': ['x^2', 'x'],
    '牛顿法': ['x^2-2', 'x', '1', '5'],
    '中值定理': ['x^2', 'x', '0', '1'],
    '弧长': ['x', 'x', '0', '1'],
    '体积(圆盘)': ['x', 'x', '0', '1'],
    '体积(柱壳)': ['x', 'x', '0', '1'],
    '曲面面积': ['x', 'x', '0', '1'],
    '偏导数': ['x^2 y', 'x'],
    '二重积分': ['x*y', 'x', 'y', '0', '1', '0', '1'],
    '隐函数求导': ['x^2+y^2-1', 'x'],
    '参数方程求导': ['cos(t)', 'sin(t)'],
    '极坐标面积': ['1+cos(theta)', 'theta', '0', 'pi'],
    '极坐标斜率': ['1+cos(theta)', 'theta'],
    '级数敛散性': ['1/n^2', 'n'],
    '收敛半径': ['1/n', 'n'],
    '幂级数(系数cₙ)': ['1/n', 'n', '0'],
    '微分方程求解': ["f'(x)+f(x)-exp(x)", 'x'],
    '数值积分': ['sin(x)', 'x', '0', 'pi', '4'],
    '复数化简': ['(1+2i)/(3-i)'],
    '极坐标形式': ['1+i'],
    '复数求根': ['z^3-1', 'z'],
    '棣莫弗高次幂': ['1+i', '3'],
    '行列式': ['\\begin{bmatrix}1&2\\\\3&4\\end{bmatrix}'],
    '秩': ['[[1,2],[3,4]]'],
    '求逆': ['[[1,2],[3,4]]'],
    '行最简': ['[[1,2],[3,4]]'],
    '特征值': ['[[1,2],[3,4]]'],
    'SVD': ['[[1,2],[3,4]]'],
    'Schur': ['[[1,2],[3,4]]'],
    'LU': ['[[1,2],[3,4]]'],
    '对角化': ['[[1,2],[3,4]]'],
    '正交化': ['[[1,2],[3,4]]'],
    '相关性': ['[[1,2],[3,4]]'],
    '向量投影': ['[[1,2]]', '[[1,0],[0,1]]'],
    '矩阵加法': ['[[1,2],[3,4]]', '[[5,6],[7,8]]'],
    '矩阵减法': ['[[1,2],[3,4]]', '[[5,6],[7,8]]'],
    '矩阵乘法': ['[[1,2],[3,4]]', '[[5,6],[7,8]]'],
}


def _notebook_key(label, latex):
    if label == '定积分':
        return '定积分@structure' if latex.startswith(r'\int_') else '定积分@operator'
    if label == '极限':
        return '极限@structure' if latex.startswith(r'\lim') else '极限@operator'
    return label


def test_actual_notebook_templates_run_through_auto():
    for key in parse_keys(INDEX):
        data = key['data']
        if 'latex' not in data:
            continue
        label = key['label']
        name = _notebook_key(label, data['latex'])
        if label in NOTEBOOK_TOKENS or data['latex'] in NOTEBOOK_TOKENS:
            continue
        assert name in NOTEBOOK_FILL, key
        assert name in NOTEBOOK_VALUES, key
        values = NOTEBOOK_VALUES[name]
        filled = fill_placeholders(data['latex'], values)
        ok, result = start_compute('auto', {'operation_type': 'auto', 'expression': filled})
        assert ok, (label, filled, result)
        text = result if isinstance(result, str) else str(result)
        assert '$' not in text and r'\placeholder' not in text and r'\square' not in text

    for key in parse_keys(INDEX):
        latex = key['data'].get('latex')
        if not latex:
            continue
        assert latex.count('{') == latex.count('}'), key
        assert latex.count(r'\begin{') == latex.count(r'\end{'), key


# ---------------------------------------------------------------------------
# Matrix panel palette (matrix input only)
# ---------------------------------------------------------------------------

PANEL_TOKENS = {
    'x', 'y', 'z', 'a', 'b', 'c', 'π', 'e',
    '+', '−', '×', '÷', '幂', '下标', '分数', '( )', '[ ]', '=',
}

PANEL_MATRIX_FILL = {
    '2×2 矩阵': [1, 2, 3, 4],
    '3×3 矩阵': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    '2×3 矩阵': [1, 2, 3, 4, 5, 6],
    '行向量': [1, 2, 3],
}


def test_notebook_template_braces_are_balanced():
    for key in parse_keys(INDEX):
        latex = key['data'].get('latex')
        if not latex:
            continue
        assert latex.count('{') == latex.count('}'), key
        assert latex.count(r'\begin{') == latex.count(r'\end{'), key


def test_matrix_panel_keys_are_classified_and_matrices_parse():
    keys = parse_keys(PANEL)
    assert len(keys) == 22
    for key in keys:
        data = key['data']
        assert 'latex' in data, key
        assert 'op' not in data, key
        if key['label'] in PANEL_TOKENS:
            continue
        values = PANEL_MATRIX_FILL.get(key['label'])
        assert values is not None, key
        filled = fill_placeholders(data['latex'], values)
        response = app.test_client().post('/api/matrix_analyze', json={'expression': filled})
        payload = response.get_json()
        assert response.status_code == 200
        assert not payload.get('error'), (key['label'], payload)
        assert 'dimensions' in payload


# ---------------------------------------------------------------------------
# Matrix lab palette (executable operations)
# ---------------------------------------------------------------------------

LAB_KEYS = {
    '转置', '求逆', '行最简', '行列式', '秩', '迹',
    '交换行', '行相加', '缩放行', '交换列', '列相加', '缩放列',
    '删除行', '删除列', '子矩阵', '数乘', '幂', '加矩阵', '减矩阵', '乘矩阵',
}


def _lab_operation(key):
    data = key['data']
    ins = data.get('insert')
    assert ins, key
    if key['label'] in ('加矩阵', '减矩阵', '乘矩阵'):
        ins = fill_placeholders(ins, [5, 6, 7, 8])
    return ins


def test_matrix_lab_keys_are_executable():
    keys = parse_keys(LAB)
    assert {key['label'] for key in keys} == LAB_KEYS
    client = app.test_client()
    for key in keys:
        op = _lab_operation(key)
        converted = toggle_form(op)
        for candidate in (op, converted):
            payload = client.post('/api/matrix_cal', json={
                'expression': '[[1,2],[3,4]]',
                'operations': candidate,
                'record_all_steps': 'True',
            }).get_json()
            assert payload.get('success') is True, (key['label'], candidate, payload)
            assert payload.get('steps', '').startswith(r'\begin{align}')
            assert '$' not in payload.get('steps', '')
            assert r'\placeholder' not in payload.get('steps', '')


def test_matrix_lab_controls_have_math_labels():
    for key in parse_keys(LAB):
        assert key['data'].get('latex'), key
        assert key['data'].get('insert'), key
        assert key['data']['latex'].count('{') == key['data']['latex'].count('}'), key


# ---------------------------------------------------------------------------
# data-op mode-switch keys + palette token insert strings
# ---------------------------------------------------------------------------

DATA_OP_KEYS = {'linear-solver', 'linear-system', 'transform-1', 'transform-2'}


def test_index_data_op_keys_are_present_and_preset_by_the_form():
    """End-to-end: the four mode-switch palette keys must exist in the rendered
    page and their data-op values must be real operation-type options, because
    clicking one sets ``<select class="operation-type">.value = data-op``."""
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    html = response.get_data(as_text=True)

    keys = parse_keys(INDEX)
    data_op_keys = [key for key in keys if 'op' in key['data']]
    assert {key['data']['op'] for key in data_op_keys} == DATA_OP_KEYS
    assert len(data_op_keys) == 4
    selector_source = (ROOT / 'static' / 'script.js').read_text(encoding='utf-8')
    for key in data_op_keys:
        op = key['data']['op']
        assert f'data-op="{op}"' in html, op
        # These are mode-switch-only keys: no latex template is inserted.
        assert 'latex' not in key['data'], key
        # The palette handler assigns sel.value = op; the value must be a real
        # <option> in the generated selector, otherwise it is silently ignored.
        assert f'<option value="{op}">' in selector_source, op
        # And the backend must actually be able to route that operation type.
        assert op in SAMPLES, op


def _assert_insert_token_is_valid(token):
    """A TOKENS member is a literal fragment inserted into the math field."""
    assert isinstance(token, str) and token.strip(), token
    assert '<' not in token and '>' not in token and '&' not in token, token
    assert r'\placeholder' not in token, token
    assert token.count('{') == token.count('}'), token
    assert token.count(r'\begin{') == token.count(r'\end{'), token


def _assert_key_insert_latex_is_balanced(latex):
    """Token keys may carry \\placeholder{} slots (x^n, fractions, brackets);
    only the structural LaTeX invariants must hold for their data-latex."""
    assert isinstance(latex, str) and latex.strip(), latex
    assert '<' not in latex and '>' not in latex, latex
    assert latex.count('{') == latex.count('}'), latex
    assert latex.count(r'\begin{') == latex.count(r'\end{'), latex


def test_notebook_tokens_are_valid_insert_strings():
    # The classification tests `continue` past token keys; assert here that
    # every token is a legal insertion and that the table matches the page.
    for token in NOTEBOOK_TOKENS:
        _assert_insert_token_is_valid(token)
    page_tokens = set()
    for key in parse_keys(INDEX):
        data = key['data']
        if 'latex' not in data:
            continue
        if data['latex'] in NOTEBOOK_TOKENS or key['label'] in NOTEBOOK_TOKENS:
            _assert_key_insert_latex_is_balanced(data['latex'])
            page_tokens.add(data['latex'] if data['latex'] in NOTEBOOK_TOKENS
                            else key['label'])
    assert page_tokens == NOTEBOOK_TOKENS, sorted(NOTEBOOK_TOKENS ^ page_tokens)


def test_panel_tokens_are_valid_insert_strings():
    for token in PANEL_TOKENS:
        _assert_insert_token_is_valid(token)
    page_tokens = set()
    for key in parse_keys(PANEL):
        if key['label'] in PANEL_TOKENS:
            _assert_key_insert_latex_is_balanced(key['data'].get('latex', ''))
            page_tokens.add(key['label'])
    assert page_tokens == PANEL_TOKENS, sorted(PANEL_TOKENS ^ page_tokens)
