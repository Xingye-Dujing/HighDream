# pylint: disable=redefined-outer-name,unused-argument
"""Endpoint/API contract tests for every product page and API family."""

import importlib
import time

import pytest

from app import app
from utils import snapshot_store

api_module = importlib.import_module('routes.api')


@pytest.fixture
def client():
    return app.test_client()


@pytest.fixture
def manual_sessions():
    api_module._manual_sessions.clear()
    yield
    api_module._manual_sessions.clear()


@pytest.fixture
def temp_snapshot_db(monkeypatch, tmp_path):
    monkeypatch.setattr(snapshot_store, 'DB_DIR', tmp_path)
    monkeypatch.setattr(snapshot_store, 'DB_PATH', tmp_path / 'snapshots.db')
    snapshot_store._local.conn = None
    yield
    conn = getattr(snapshot_store._local, 'conn', None)
    if conn is not None:
        conn.close()
    snapshot_store._local.conn = None


def _wait(client, task_id, timeout=8.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        payload = client.post('/api/task_status', json={'task_id': task_id}).get_json()
        if payload.get('success') and payload['task']['status'] in ('completed', 'failed'):
            return payload['task']
        time.sleep(0.02)
    raise AssertionError(f'task did not finish: {task_id}')


def test_all_get_pages_return_200(client):
    routes = []
    for rule in app.url_map.iter_rules():
        if 'GET' not in rule.methods or rule.arguments:
            continue
        if rule.rule.startswith('/static') or rule.rule == '/api/stream':
            continue
        routes.append(rule.rule)
    assert routes
    for path in sorted(set(routes)):
        response = client.get(path)
        assert response.status_code == 200, path
        assert response.data, path


API_ENDPOINTS = {
    '/api/parse', '/api/compute', '/api/convert_form', '/api/steps', '/api/plot',
    '/api/stream', '/api/task_status', '/api/matrix_analyze', '/api/matrix_cal',
    '/api/get_process', '/api/render_latex', '/api/sympy_calculate',
    '/api/sympy_simplify', '/api/sympy_execute', '/api/manual_start',
    '/api/manual_step', '/api/manual_undo', '/api/manual_fallback',
    '/api/manual_reset', '/api/method_tree_start', '/api/method_tree_status',
    '/api/method_tree_cancel', '/api/method_tree_respond', '/api/method_tree_path',
    '/api/snapshot_save', '/api/snapshot_list', '/api/snapshot_get',
    '/api/snapshot_delete',
}


def test_api_route_surface_is_explicitly_known():
    actual = {rule.rule for rule in app.url_map.iter_rules()
              if rule.rule.startswith('/api/')}
    assert actual == API_ENDPOINTS


@pytest.mark.parametrize('path', sorted(API_ENDPOINTS - {
    '/api/stream', '/api/snapshot_list', '/api/snapshot_get',
}))
def test_post_api_empty_body_does_not_crash(client, path):
    response = client.post(path, json={})
    assert response.status_code == 200, path
    assert response.is_json, path
    payload = response.get_json()
    assert isinstance(payload, dict), path


def test_matrix_pages_accept_and_reject_expression_forms(client):
    valid = client.get('/matrix_analysis?expression=' + r'\begin{bmatrix}1&2\\3&4\end{bmatrix}')
    assert valid.status_code == 200
    assert b'[[1,2],[3,4]]' in valid.data
    invalid = client.get('/matrix_analysis?expression=' + r'\begin{bmatrix}1&2\\3\end{bmatrix}')
    assert invalid.status_code == 200
    assert b'preview-error' in invalid.data

    posted = client.post('/matrix_analysis', data={'expression': '[[1,2],[3,4]]'})
    assert posted.status_code == 200 and b'[[1,2],[3,4]]' in posted.data

    lab = client.get('/matrix_lab?expression=[[1,2],[3,4]]')
    assert lab.status_code == 200
    assert b'matrix-lab-start' in lab.data and b'[[1,2],[3,4]]' in lab.data
    lab_bad = client.get('/matrix_lab?expression=[[1,2],[3]]')
    assert lab_bad.status_code == 200
    assert b'preview-error' in lab_bad.data
    lab_post = client.post('/matrix_lab', data={'expression': '[[1,2],[3,4]]'})
    assert lab_post.status_code == 200 and b'[[1,2],[3,4]]' in lab_post.data


def test_compute_task_lifecycle_and_sse(client):
    response = client.post('/api/compute', json={
        'operation_type': 'auto', 'expression': 'x+x',
    }).get_json()
    assert response['success'] is True
    task = _wait(client, response['task_id'])
    assert task['status'] == 'completed', task
    assert task['result']

    stream = client.get('/api/stream?task_id=' + response['task_id'])
    assert stream.status_code == 200
    assert stream.mimetype == 'text/event-stream'
    body = stream.get_data(as_text=True)
    assert 'event: completed' in body


def test_compute_task_for_representative_families(client):
    cases = [
        ('diff', {'expression': 'x^2', 'variable': 'x'}),
        ('integral', {'expression': 'sin(x)', 'variable': 'x'}),
        ('limit', {'expression': 'sin(x)/x', 'variable': 'x', 'point': '0', 'direction': 'both'}),
        ('det', {'expression': '[[1,2],[3,4]]', 'type': '1'}),
        ('tangent', {'expression': 'x^2', 'variable': 'x', 'point': '1'}),
        ('complex_simplify', {'expression': '(1+2i)/(3-i)'}),
        ('expr', {'expression': 'x^2-1', 'max_depth': '3', 'sort_strategy': 'complexity'}),
    ]
    for op, extra in cases:
        response = client.post('/api/compute', json={'operation_type': op, **extra}).get_json()
        assert response['success'] is True, (op, response)
        task = _wait(client, response['task_id'])
        assert task['status'] == 'completed', (op, task)
        assert task['result'] is not None, op


def test_compute_failure_task_and_missing_task_status(client):
    response = client.post('/api/compute', json={
        'operation_type': 'not-an-operation', 'expression': 'x',
    }).get_json()
    task = _wait(client, response['task_id'])
    assert task['status'] == 'failed'
    assert task.get('error')

    missing = client.post('/api/task_status', json={'task_id': 'missing'}).get_json()
    assert missing['success'] is False

    stream = client.get('/api/stream?task_id=missing')
    assert 'event: failed' in stream.get_data(as_text=True)


def test_steps_and_plot_endpoints(client):
    for op, extra in [
        ('diff', {'expression': 'x^2', 'variable': 'x'}),
        ('integral', {'expression': 'x^2', 'variable': 'x'}),
        ('limit', {'expression': 'sin(x)/x', 'variable': 'x', 'point': '0'}),
    ]:
        payload = client.post('/api/steps', json={'operation_type': op, **extra}).get_json()
        assert payload['success'] is True, (op, payload)
        assert payload['steps']

    bad = client.post('/api/steps', json={'operation_type': 'det', 'expression': 'x'}).get_json()
    assert bad['success'] is False

    plot = client.post('/api/plot', json={
        'expression': 'x^2', 'variable': 'x', 'xmin': -2, 'xmax': 2, 'samples': 20,
    }).get_json()
    assert plot['success'] is True, plot
    assert len(plot['points']) == 20
    bad_plot = client.post('/api/plot', json={'expression': '((('}).get_json()
    assert bad_plot['success'] is False


def test_parse_render_and_convert_endpoints(client):
    parse = client.post('/api/parse', json={'expression': 'x^2', 'operation_type': 'expr'}).get_json()
    assert parse['success'] is True and 'x^{2}' in parse['result']

    matrix = client.post('/api/parse', json={
        'expression': r'\begin{bmatrix}1&2\\3&4\end{bmatrix}', 'operation_type': 'matrix',
    }).get_json()
    assert matrix['success'] is True and 'matrix' in matrix['result']

    system = client.post('/api/parse', json={
        'expression': 'x+y=2\nx-y=0', 'operation_type': 'linear-system',
    }).get_json()
    assert system['success'] is True and 'cases' in system['result']

    unfilled = client.post('/api/parse', json={
        'expression': r'\frac{}{}', 'operation_type': 'expr',
    }).get_json()
    assert unfilled['success'] is False

    # Regression: operator input used to fall into SymPy's implicit-multiplication
    # fallback and preview as "di f f(x^2, x)" / "a e g i n r t..." garbage.
    diff_preview = client.post('/api/parse', json={
        'expression': 'diff(x^2,x)', 'operation_type': 'auto',
    }).get_json()
    assert diff_preview['success'] is True
    assert r'\operatorname{diff}' in diff_preview['result']
    integral_preview = client.post('/api/parse', json={
        'expression': 'integrate(x^2,x)', 'operation_type': 'auto',
    }).get_json()
    assert integral_preview['success'] is True
    assert r'\operatorname{integrate}' in integral_preview['result']

    rendered = client.post('/api/render_latex', json={'expression': 'x^2'}).get_json()
    assert rendered['success'] is True and 'x^{2}' in rendered['latex']
    assert client.post('/api/render_latex', json={'expression': '((('}).get_json()['success'] is False

    assert client.post('/api/convert_form', json={'expression': 'int(x^2,x)'}).get_json()['result'].startswith(r'\int')
    assert client.post('/api/convert_form', json={
        'expression': r'\begin{bmatrix}1&2\\3&4\end{bmatrix}',
    }).get_json()['result'] == '[[1,2],[3,4]]'
    assert client.post('/api/convert_form', json={'expression': '[[1,2],[3,4]]'}).get_json()['result'] == '[[1,2],[3,4]]'
    assert client.post('/api/convert_form', json={'expression': r'\frac{}{}'}).get_json()['result'] == r'\frac{}{}'


def test_sympy_endpoints(client):
    diff = client.post('/api/sympy_calculate', json={
        'expression': 'x^2', 'operation_type': 'diff', 'variable': 'x',
    }).get_json()
    assert diff['success'] is True and diff['result']
    # Regression: Symbol(variable, real=True) is a *different* symbol than the
    # assumption-free symbols sympify produces, so every derivative used to
    # come back as "0" and every integral as "x*x".
    assert diff['result'] == '2*x'
    partial = client.post('/api/sympy_calculate', json={
        'expression': 'x*y', 'operation_type': 'diff', 'variable': 'x',
    }).get_json()
    assert partial['result'] == 'y'
    integral = client.post('/api/sympy_calculate', json={
        'expression': 'x', 'operation_type': 'integral', 'variable': 'x',
    }).get_json()
    assert integral['result'] == 'x**2/2'
    limit = client.post('/api/sympy_calculate', json={
        'expression': 'sin(x)/x', 'operation_type': 'limit', 'variable': 'x',
        'point': '0', 'direction': '+-',
    }).get_json()
    assert limit['result'] == '1'
    for op, extra in [
        ('integral', {}), ('limit', {'point': '0', 'direction': '+'}),
    ]:
        payload = client.post('/api/sympy_calculate', json={
            'expression': 'sin(x)', 'operation_type': op, 'variable': 'x', **extra,
        }).get_json()
        assert payload['success'] is True, (op, payload)
    assert client.post('/api/sympy_calculate', json={'expression': '((('}).get_json()['success'] is False

    simplified = client.post('/api/sympy_simplify', json={'expression': 'x+x'}).get_json()
    assert simplified['success'] is True and simplified['result']
    assert client.post('/api/sympy_simplify', json={'expression': '((('}).get_json()['success'] is False

    assert client.post('/api/sympy_execute', json={'code': '1+1'}).get_json()['success'] is True
    assert client.post('/api/sympy_execute', json={'code': ''}).get_json()['success'] is False
    assert client.post('/api/sympy_execute', json={'code': '1/0'}).get_json()['success'] is False


def test_matrix_api_endpoints(client):
    analyzed = client.post('/api/matrix_analyze', json={'expression': '[[1,2],[3,4]]'}).get_json()
    assert analyzed['dimensions'] == '2×2'
    assert analyzed['determinant'] == '-2'
    invalid = client.post('/api/matrix_analyze', json={'expression': '[[1,2],[3]]'}).get_json()
    assert invalid['success'] is False
    # Regression: non-string payloads used to raise TypeError -> HTTP 500.
    for bad in (None, 123, [], True, {'a': 1}):
        response = client.post('/api/matrix_analyze', json={'expression': bad})
        assert response.status_code == 200, bad
        assert response.get_json()['success'] is False, bad

    for op_type in ('det_1', 'det_2', 'rank', 'eigen', 'svd'):
        payload = client.post('/api/get_process', json={
            'type': op_type, 'expression': '[[1,2],[3,4]]',
        }).get_json()
        assert payload['success'] is True, (op_type, payload)
        assert payload['process']
    bad_process = client.post('/api/get_process', json={
        'type': 'det_1', 'expression': '[[1,2],[3]]',
    }).get_json()
    assert bad_process['success'] is False

    cal = client.post('/api/matrix_cal', json={
        'expression': '[[1,2],[3,4]]', 'operations': 'transpose', 'record_all_steps': 'True',
    }).get_json()
    assert cal['success'] is True and cal['steps'].startswith(r'\begin{align}')
    bad_cal = client.post('/api/matrix_cal', json={
        'expression': '[[1,2],[3,4]]', 'operations': 'nope', 'record_all_steps': 'True',
    }).get_json()
    assert bad_cal['success'] is False and bad_cal['error']


def test_manual_rule_selection_api(client, manual_sessions):
    missing = client.post('/api/manual_step', json={'session_id': 'missing'}).get_json()
    assert missing['success'] is False
    for endpoint in ('/api/manual_undo', '/api/manual_fallback'):
        payload = client.post(endpoint, json={'session_id': 'missing'}).get_json()
        assert payload['success'] is False
    assert client.post('/api/manual_reset', json={'session_id': 'missing'}).get_json()['success'] is True

    bad = client.post('/api/manual_start', json={'domain': 'nope', 'expression': 'x'}).get_json()
    assert bad['success'] is False

    for domain, expression, extra in [
        ('diff', 'x**2 + sin(x)', {}),
        ('integral', 'x*exp(x**2)', {}),
        ('limit', 'sin(3*x)/x', {'point': '0', 'direction': '+'}),
    ]:
        started = client.post('/api/manual_start', json={
            'domain': domain, 'expression': expression, 'variable': 'x', **extra,
        }).get_json()
        assert started['success'] is True, (domain, started)
        session = started['session_id']
        state = started['state']
        assert state['steps']
        rules = state.get('applicable_rules') or []
        assert rules, (domain, state)
        stepped = client.post('/api/manual_step', json={
            'session_id': session, 'rule_name': rules[0]['name'],
        }).get_json()
        assert stepped['success'] is True, (domain, stepped)
        undone = client.post('/api/manual_undo', json={'session_id': session}).get_json()
        assert undone['success'] is True
        fallback = client.post('/api/manual_fallback', json={'session_id': session}).get_json()
        assert fallback['success'] is True, (domain, fallback)
        assert client.post('/api/manual_reset', json={'session_id': session}).get_json()['success'] is True
        assert client.post('/api/manual_undo', json={'session_id': session}).get_json()['success'] is False


def test_method_tree_api(client):
    assert client.post('/api/method_tree_start', json={}).get_json()['success'] is False
    assert client.post('/api/method_tree_start', json={'domain': 'diff'}).get_json()['success'] is False

    started = client.post('/api/method_tree_start', json={
        'domain': 'diff', 'expression': 'x^2', 'variable': 'x',
        'max_depth': 2, 'max_nodes': 20, 'time_limit_seconds': 5,
    }).get_json()
    assert started['success'] is True
    task_id = started['task_id']
    status = client.post('/api/method_tree_status', json={'task_id': task_id}).get_json()
    assert status['success'] is True
    cancelled = client.post('/api/method_tree_cancel', json={'task_id': task_id}).get_json()
    assert cancelled['success'] is True
    responded = client.post('/api/method_tree_respond', json={
        'task_id': task_id, 'accepted': True,
    }).get_json()
    assert responded['success'] is True
    assert client.post('/api/method_tree_status', json={'task_id': 'missing'}).get_json()['success'] is False

    path = client.post('/api/method_tree_path', json={
        'domain': 'diff', 'expression': 'x^2', 'variable': 'x', 'rule_path': [],
    }).get_json()
    assert path['success'] is True, path


def test_manual_sessions_are_pruned(monkeypatch):
    from routes import manual_api  # pylint: disable=import-outside-toplevel

    manual_api._manual_sessions.clear()  # pylint: disable=protected-access
    now = time.monotonic()
    # 5 expired entries + cap+5 fresh ones; prune must remove both classes.
    for i in range(5):
        manual_api._manual_sessions[f'old{i}'] = (object(), now - 10_000)  # pylint: disable=protected-access
    for i in range(manual_api._MANUAL_SESSION_MAX + 5):  # pylint: disable=protected-access
        # s0 is the oldest entry; the overflow evicts the oldest first.
        manual_api._manual_sessions[f's{i}'] = (  # pylint: disable=protected-access
            object(), now - (manual_api._MANUAL_SESSION_MAX + 5 - i))
    manual_api._prune_manual_sessions()  # pylint: disable=protected-access
    sessions = manual_api._manual_sessions  # pylint: disable=protected-access
    assert len(sessions) == manual_api._MANUAL_SESSION_MAX  # pylint: disable=protected-access
    assert all(not key.startswith('old') for key in sessions)
    assert 's0' not in sessions and 's5' in sessions
    sessions.clear()


def test_snapshot_crud_api(client, temp_snapshot_db):
    empty = client.post('/api/snapshot_save', json={'cells': []}).get_json()
    assert empty['success'] is False
    too_big = client.post('/api/snapshot_save', json={
        'cells': [{'type': 'code', 'content': 'x' * (snapshot_store.MAX_SNAPSHOT_BYTES + 1)}],
    }).get_json()
    assert too_big['success'] is False
    assert '2 MB' in too_big['error']
    saved = client.post('/api/snapshot_save', json={
        'cells': [{'type': 'code', 'content': 'x^2'}], 'title': 'first',
    }).get_json()
    assert saved['success'] is True
    snapshot_id = saved['snapshot']['id']

    listed = client.get('/api/snapshot_list?limit=10').get_json()
    assert listed['success'] is True
    assert [item['id'] for item in listed['snapshots']] == [snapshot_id]

    full = client.get('/api/snapshot_get?id=' + str(snapshot_id)).get_json()
    assert full['success'] is True
    assert 'x^2' in full['snapshot']['payload']
    assert client.get('/api/snapshot_get?id=bad').get_json()['success'] is False
    assert client.get('/api/snapshot_get?id=999999').get_json()['success'] is False

    assert client.post('/api/snapshot_delete', json={'id': snapshot_id}).get_json()['success'] is True
    assert client.post('/api/snapshot_delete', json={'id': snapshot_id}).get_json()['success'] is False
    assert client.get('/api/snapshot_list?limit=bad').get_json()['success'] is True


def test_snapshot_store_prunes_to_max(temp_snapshot_db):
    created = [snapshot_store.create_snapshot('[]', f's{i}') for i in range(205)]
    rows = snapshot_store.list_snapshots(1000)
    assert len(rows) == snapshot_store.MAX_SNAPSHOTS
    assert rows[0]['id'] == created[-1]['id']
    assert snapshot_store.get_snapshot(created[0]['id']) is None
