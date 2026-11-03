# pylint: disable=cyclic-import
"""Matrix-page pipeline: WYSIWYG operation keys and server-side validation."""

# The test imports the Flask app, which pulls in the compute route; the route
# honestly reuses op_form helpers.

from app import app


def _client():
    return app.test_client()


def test_matrix_cal_accepts_latex_operation_commands():
    client = _client()
    for operation in [
        r'\mathrm{transpose}',
        r'\mathrm{inverse}',
        r'\mathrm{rref}',
        r'\mathrm{det}',
        r'\mathrm{rank}',
        r'\mathrm{trace}',
        r'\mathrm{swap\_rows}(0,1)',
        r'\mathrm{scale\_row}(0,2)',
        r'\mathrm{add}\left(\begin{bmatrix}5&6\\7&8\end{bmatrix}\right)',
    ]:
        data = client.post('/api/matrix_cal', json={
            'expression': '[[1,2],[3,4]]',
            'operations': operation,
            'record_all_steps': 'True',
        }).get_json()
        assert data['success'] is True, (operation, data)
        assert data['steps'].startswith(r'\begin{align}')


def test_matrix_analyze_rejects_invalid_input_gracefully():
    data = _client().post('/api/matrix_analyze', json={
        'expression': r'\begin{bmatrix}1&2\\3\end{bmatrix}',
    }).get_json()
    assert data.get('success') is False
    assert data.get('error')


def test_matrix_analysis_page_accepts_bmatrix_url():
    client = _client()
    response = client.get('/matrix_analysis?expression=' + r'\begin{bmatrix}1&2\\3&4\end{bmatrix}')
    assert response.status_code == 200
    assert b'[[1,2],[3,4]]' in response.data


def test_matrix_lab_page_without_expression_has_start_form():
    response = _client().get('/matrix_lab')
    assert response.status_code == 200
    assert b'matrix-lab-start' in response.data


def test_matrix_cal_reports_operation_error():
    data = _client().post('/api/matrix_cal', json={
        'expression': '[[1,2],[3,4]]',
        'operations': 'unknown_op',
        'record_all_steps': 'True',
    }).get_json()
    assert data['success'] is False
    assert 'unknown_op' in data['error']
