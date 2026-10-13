"""Tests for the structured per-step solver output backing step-linking."""

from core.step_tree import structured_solve


def test_diff_structure_and_variables():
    result = structured_solve('diff', {'expression': 'x**2*sin(x)', 'variable': 'x'})
    assert result['success'] and result['steps']
    for step in result['steps']:
        assert {'latex', 'reason', 'subtasks'} <= set(step)
    subtasks = [s for step in result['steps'] for s in step['subtasks']]
    assert subtasks
    assert all(s['op'] == 'diff' and s['variable'] == 'x' for s in subtasks)


def test_integral_structure():
    result = structured_solve('integral', {'expression': 'sin(x)*cos(x)', 'variable': 'x'})
    assert result['success'] and result['steps']
    first = result['steps'][0]
    assert first['subtasks'][0]['op'] == 'integral'


def test_limit_structure():
    result = structured_solve('limit', {'expression': 'sin(x)/x', 'variable': 'x', 'point': '0'})
    assert result['success'] and result['steps']


def test_unsupported_operation_returns_error():
    result = structured_solve('bogus', {'expression': 'x'})
    assert result['success'] is False
    assert 'error' in result
