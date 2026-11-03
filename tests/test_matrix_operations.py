"""Strict contracts for every MatrixCalculator operation and its display."""

import pytest

from app import app

BASE = '[[1,2],[3,4]]'

OPERATIONS = [
    'transpose', 'inverse', 'rref', 'ref', 'det', 'rank', 'trace',
    'add [[5,6],[7,8]]', 'subtract [[5,6],[7,8]]', 'multiply [[5,6],[7,8]]',
    'scale 2', 'power 2',
    'swap_rows 0 1', 'swap_cols 0 1',
    'add_rows 0 1 1', 'add_cols 0 1 1',
    'scale_row 0 3', 'scale_col 0 3',
    'insert_row 1 9 9', 'insert_col 1 9 9',
    'remove_row 0', 'remove_col 0', 'submatrix 0 1 0 1',
]


def _run(operation, record=True):
    return app.test_client().post('/api/matrix_cal', json={
        'expression': BASE,
        'operations': operation,
        'record_all_steps': 'True' if record else 'False',
    }).get_json()


@pytest.mark.parametrize('operation', OPERATIONS)
def test_every_matrix_operation_succeeds_with_valid_latex(operation):
    payload = _run(operation)
    assert payload['success'] is True, (operation, payload)
    steps = payload['steps']
    assert steps.startswith(r'\begin{align}') and steps.endswith(r'\end{align}')
    assert '$' not in steps
    assert r'\placeholder' not in steps
    assert r'\square' not in steps
    assert steps.count('{') == steps.count('}')
    assert r'\begin{matrix}' in steps or operation in ('det', 'rank', 'trace')


def test_matrix_operation_results_are_correct():
    cases = {
        'transpose': r'1 & 3\\2 & 4',
        'add [[5,6],[7,8]]': r'6 & 8\\10 & 12',
        'subtract [[5,6],[7,8]]': r'-4 & -4\\-4 & -4',
        'multiply [[5,6],[7,8]]': r'19 & 22\\43 & 50',
        'inverse': r'-2 & 1\\\frac{3}{2} & - \frac{1}{2}',
        'det': '-2',
        'rank': '2',
        'trace': '5',
        'scale_row 0 3': r'3 & 6',
        'swap_rows 0 1': r'3 & 4\\1 & 2',
    }
    for operation, expected in cases.items():
        assert expected in _run(operation)['steps'], operation


def test_final_only_mode_omits_intermediate_matrix():
    payload = _run('transpose', record=False)
    assert payload['success'] is True
    assert '初始矩阵' not in payload['steps']
    assert r'\begin{matrix}' in payload['steps']


@pytest.mark.parametrize('operation', ['', '   ', 'unknown', 'swap_rows 0', 'scale_row x y'])
def test_invalid_matrix_operation_is_rejected(operation):
    payload = _run(operation)
    assert payload['success'] is False
    assert payload.get('error')


def test_singular_inverse_is_rejected_with_friendly_message():
    payload = app.test_client().post('/api/matrix_cal', json={
        'expression': '[[1,2],[2,4]]',
        'operations': 'inverse',
        'record_all_steps': 'True',
    }).get_json()
    assert payload['success'] is False
    assert '不可逆' in payload['error']
