"""Exhaustive operation-contract tests.

Every operation type reachable from the universal box, the operation selector,
or the applied/matrix handlers must have a known-good sample here.  If a new
operation is added and this map is not extended, the suite fails; that is the
guard against silent loss of coverage.
"""

import ast
import pathlib
import re

import pytest

from core.operator_registry import _MATRIX_LATEX_OPS
from domains.applied import HANDLERS
from routes.compute_service import _AUTO_FUNCS, _build_op_data, start_compute

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Known-good samples for every distinct operation type
# ---------------------------------------------------------------------------

SAMPLES = {
    # universal dispatchers / expression
    'auto': {'expression': 'x^2'},
    'expr': {'expression': 'x^2 - 1'},
    'calculus': {'expression': r'\frac{d}{dx}(x^2)', 'variable': 'x'},
    # core calculus
    'diff': {'expression': 'sin(x)*x', 'variable': 'x'},
    'integral': {'expression': 'sin(x)', 'variable': 'x'},
    'limit': {'expression': 'sin(x)/x', 'variable': 'x', 'point': '0', 'direction': 'both'},
    # matrix
    'operations': {'expression': '[[1,2],[3,4]]\n[[5,6],[7,8]]', 'operations': '+'},
    'matrix_symbolic': {'expression': '(A+B)*(A-B)'},
    'invert': {'expression': '[[1,2],[3,4]]'},
    'ref': {'expression': '[[1,2],[3,4]]', 'target_form': 'rref'},
    'det': {'expression': '[[1,2],[3,4]]', 'type': '1'},
    'trace': {'expression': '[[1,2],[3,4]]'},
    'pinv': {'expression': '[[1,2],[3,4]]'},
    'jordan': {'expression': '[[1,2],[3,4]]'},
    'definiteness': {'expression': '[[1,2],[2,1]]'},
    'charpoly': {'expression': '[[1,2],[3,4]]'},
    'nullspace': {'expression': '[[1,2],[3,4]]'},
    'colspace': {'expression': '[[1,2],[3,4]]'},
    'rowspace': {'expression': '[[1,2],[3,4]]'},
    'rank': {'expression': '[[1,2],[3,4]]'},
    'LU': {'expression': '[[1,2],[3,4]]'},
    'orthogonal': {'expression': '[[1,2],[3,4]]'},
    'diag': {'expression': '[[1,2],[3,4]]'},
    'eigen': {'expression': '[[1,2],[3,4]]'},
    'singular': {'expression': '[[1,2],[3,4]]'},
    'svd': {'expression': '[[1,2],[3,4]]'},
    'schur': {'expression': '[[1,2],[3,4]]'},
    'dependence': {'expression': '[[1,2],[3,4]]'},
    'projection': {'expression': '[[1,2]]\n[[1,0],[0,1]]'},
    'linear-system': {'expression': 'x+y=2\nx-y=0'},
    'linear-solver': {'expression': '[[1,2],[3,4]]\n[[1],[1]]'},
    'transform-1': {'expression': '[[1,0],[0,1]]\n[[1,1],[0,1]]', 'type': 'basis_change'},
    'transform-2': {'expression': '[[2,0],[0,2]]\n[[1,0],[0,1]]', 'type': 'find_matrix'},
    # differentiation applications
    'tangent': {'expression': 'x^2', 'variable': 'x', 'point': '1'},
    'linearization': {'expression': 'x^2', 'variable': 'x', 'point': '1'},
    'critical_points': {'expression': 'x^2-2*x', 'variable': 'x'},
    'newton': {'expression': 'x^2-2', 'variable': 'x', 'point': '1', 'order': '5'},
    'bisection': {'expression': 'x^2-2', 'variable': 'x', 'lower': '0', 'upper': '2', 'order': '8'},
    'secant': {'expression': 'x^2-2', 'variable': 'x', 'point': '1', 'x1': '2', 'order': '5'},
    'fixed_point': {'expression': 'cos(x)', 'variable': 'x', 'point': '1', 'order': '5'},
    'mean_value': {'expression': 'x^2', 'variable': 'x', 'lower': '0', 'upper': '2'},
    # integral applications
    'arc_length': {'expression': 'x', 'variable': 'x', 'lower': '0', 'upper': '1'},
    'volume_disk': {'expression': 'x', 'variable': 'x', 'lower': '0', 'upper': '1'},
    'volume_shell': {'expression': 'x', 'variable': 'x', 'lower': '0', 'upper': '1'},
    'surface_area': {'expression': 'x', 'variable': 'x', 'lower': '0', 'upper': '1'},
    'iterated_integral': {'expression': 'x*y', 'variable': 'x', 'variable2': 'y',
                          'lower': '0', 'upper': '1', 'lower2': '0', 'upper2': '1'},
    'triple_integral': {'expression': 'x*y*z', 'variable': 'x', 'variable2': 'y',
                        'variable3': 'z', 'lower': '0', 'upper': '1',
                        'lower2': '0', 'upper2': '1', 'lower3': '0', 'upper3': '1'},
    'changeorder': {'expression': 'x', 'variable': 'x', 'variable2': 'y',
                    'lower': '0', 'upper': '1', 'lower2': '0', 'upper2': '1'},
    'cylindrical': {'expression': '1', 'lower': '0', 'upper': '1',
                    'lower2': '0', 'upper2': '2*pi', 'lower3': '0', 'upper3': '1'},
    'spherical': {'expression': '1', 'lower': '0', 'upper': '1',
                  'lower2': '0', 'upper2': 'pi', 'lower3': '0', 'upper3': '2*pi'},
    # convergence / series
    'series_convergence': {'expression': '1/n^2', 'variable': 'n'},
    'finite_sum': {'expression': 'n', 'variable': 'n', 'lower': '1', 'upper': '3'},
    'finite_product': {'expression': 'n', 'variable': 'n', 'lower': '1', 'upper': '3'},
    'radius_of_convergence': {'expression': '1/n', 'variable': 'n'},
    'power_series': {'expression': '1/n', 'variable': 'n', 'point': '0'},
    'taylor': {'expression': 'exp(x)', 'variable': 'x', 'point': '0', 'order': '3'},
    'taylor_error': {'expression': 'exp(x)', 'variable': 'x', 'point': '0', 'order': '3'},
    'improper_integral': {'expression': '1/x^2', 'variable': 'x', 'lower': '1', 'upper': 'oo'},
    # ODE / numerical
    'ode': {'expression': "f'(x)+f(x)-exp(x)", 'variable': 'x'},
    'numerical_integration': {'expression': 'sin(x)', 'variable': 'x',
                              'lower': '0', 'upper': 'pi', 'order': '4'},
    'euler': {'expression': 'x+y', 'variable': 'x', 'point': '0', 'yvalue': '1',
              'target': '1', 'steps': '4'},
    'heun': {'expression': 'x+y', 'variable': 'x', 'point': '0', 'yvalue': '1',
             'target': '1', 'steps': '4'},
    'rk4': {'expression': 'x+y', 'variable': 'x', 'point': '0', 'yvalue': '1',
            'target': '1', 'steps': '4'},
    # multivar
    'implicit_diff': {'expression': 'x^2+y^2=1', 'variable': 'x'},
    'parametric_diff': {'expression': 'cos(t), sin(t)', 'variable': 't'},
    'polar_area': {'expression': '1+cos(theta)', 'variable': 'theta',
                   'lower': '0', 'upper': 'pi'},
    'polar_slope': {'expression': '1+cos(theta)', 'variable': 'theta'},
    'gradient': {'expression': 'x^2+y^2', 'variable': 'x'},
    'hessian': {'expression': 'x^2+y^2', 'variable': 'x'},
    'jacobian': {'expression': 'x*y|x+y', 'variable': 'x|y'},
    'directional': {'expression': 'x^2+y^2', 'direction': '1|1'},
    'curl': {'expression': 'y|-x|0', 'variable': 'x|y|z'},
    'divergence': {'expression': 'x|y|z', 'variable': 'x|y|z'},
    'lagrange': {'expression': 'x*y|x+y-2', 'variable': 'x|y'},
    # line / surface integrals
    'line_integral': {'expression': '1', 'curve': 'cos(t)|sin(t)',
                      'variable': 't', 'lower': '0', 'upper': '2*pi'},
    'line_integral2': {'expression': '-y|x', 'curve': 'cos(t)|sin(t)',
                       'variable': 't', 'lower': '0', 'upper': '2*pi'},
    'surface_integral': {'expression': '1', 'surface': '0', 'variable': 'x',
                         'variable2': 'y', 'lower': '0', 'upper': '1',
                         'lower2': '0', 'upper2': '1'},
    'flux': {'expression': '0|0|1', 'surface': '0', 'variable': 'x',
             'variable2': 'y', 'lower': '0', 'upper': '1',
             'lower2': '0', 'upper2': '1'},
    # complex
    'complex_simplify': {'expression': '(1+2i)/(3-i)'},
    'complex_polar': {'expression': '1+i'},
    'complex_roots': {'expression': 'z^3-1', 'variable': 'z'},
    'complex_nth_roots': {'expression': '1+i', 'order': '3'},
    'de_moivre': {'expression': '1+i', 'order': '3'},
}

ALL_OPS = set(SAMPLES) | set(HANDLERS) | {value[0] for value in _AUTO_FUNCS.values()} | {
    'auto', 'calculus', 'linear-system', 'linear-solver', 'transform-1', 'transform-2',
}

MUST_REJECT = ['', '   ', r'\placeholder{}', r'\frac{}{}', r'\sqrt{}', '(((']

MALFORMED = MUST_REJECT + [
    r'\frac{', '[[1,2],[3]]', 'x^', '1/0', 'oo', r'\unknown{1}',
    'sin(', '{}', '[[1,2],[3,4]',
]


def _result_latex(result):
    if isinstance(result, str):
        return [result]
    if isinstance(result, tuple):
        result = result[0]
    if isinstance(result, list):
        out = []
        for item in result:
            if isinstance(item, dict):
                out.append(str(item.get('latex', '')))
            else:
                out.append(str(item))
        return out
    return [str(result)]


def assert_latex_contract(result):
    """Every successful result must be live, renderable LaTeX."""
    pieces = _result_latex(result)
    assert pieces and any(piece.strip() for piece in pieces), result
    for latex in pieces:
        assert latex.strip(), result
        assert '\x00' not in latex
        assert '$' not in latex
        assert '\\newline' not in latex
        assert '\\placeholder' not in latex
        assert '\\square' not in latex
        assert '\\Box' not in latex
        assert '\\bigbox' not in latex
        assert not re.search(r'\\text\{[^{}]*\$', latex), latex
        assert latex.count('{') == latex.count('}'), latex
        assert len(re.findall(r'\\left(?![A-Za-z])', latex)) == len(
            re.findall(r'\\right(?![A-Za-z])', latex)), latex
        for env in ('align', 'aligned', 'cases', 'matrix', 'bmatrix', 'pmatrix', 'vmatrix'):
            assert latex.count(r'\begin{%s}' % env) == latex.count(r'\end{%s}' % env), latex
        assert not re.search(r'\\frac\s*\{\s*\}\s*\{\s*\}', latex)
        assert not re.search(r'\\sqrt\s*(?:\[[^\]]*\])?\s*\{\s*\}', latex)
        assert not latex.endswith(r'\begin{align}\end{align}')


def test_samples_cover_every_discovered_operation():
    missing = sorted(ALL_OPS - set(SAMPLES))
    assert not missing, f'add a known-good sample for: {missing}'
    extra = sorted(set(SAMPLES) - ALL_OPS)
    assert not extra, f'sample for unknown operation: {extra}'


def _dispatch_string_literals_from_source(source):
    """AST-extract every string literal compared against ``operation_type``.

    The old ``operation_type\\s*==\\s*'...'`` regex missed tuple membership
    branches (``in ('nullspace','colspace','rowspace')``) and the dynamic
    ``HANDLERS.get(...)`` dispatch.  Walking the AST of ``_dispatch_compute``
    collects every string constant in such comparisons, including the elements
    of ``in (...)`` tuples.
    """
    tree = ast.parse(source)
    func = next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == '_dispatch_compute')
    literals = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Compare)
                and isinstance(node.left, ast.Name)
                and node.left.id == 'operation_type'):
            continue
        for comparator in node.comparators:
            for sub in ast.walk(comparator):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    literals.add(sub.value)
    return literals


def test_every_explicit_dispatch_branch_has_a_sample():
    source = (ROOT / 'routes' / 'compute_service.py').read_text(encoding='utf-8')
    branches = _dispatch_string_literals_from_source(source)
    # HANDLERS.get(operation_type) dispatches dynamically: no literal appears in
    # a comparison, so add all 50 registered applied handlers explicitly.
    branches |= set(HANDLERS)
    assert len(HANDLERS) == 50
    # method_tree is an async task with no single-expression sample.
    branches.discard('method_tree')
    assert branches <= set(SAMPLES), sorted(branches - set(SAMPLES))
    assert branches <= ALL_OPS, sorted(branches - ALL_OPS)
    # Pin the branches the regex-based version could not see.
    assert {'nullspace', 'colspace', 'rowspace', 'singular'} <= branches


def test_ast_extraction_catches_tuple_branches_but_not_method_calls():
    source = (
        'def _dispatch_compute(operation_type, data):\n'
        "    if operation_type in ('nullspace', 'colspace', 'rowspace'):\n"
        '        return 1\n'
        "    if operation_type == 'det':\n"
        '        return 2\n'
        "    if operation_type.startswith('complex_'):\n"
        '        return 3\n'
    )
    assert _dispatch_string_literals_from_source(source) == {
        'nullspace', 'colspace', 'rowspace', 'det',
    }


def test_frontend_operation_options_are_tested():
    script = (ROOT / 'static' / 'script.js').read_text(encoding='utf-8')
    values = set(re.findall(r'<option value="([^"]+)"', script))
    non_ops = {
        '+', '-', 'T', '*', 'scalar_mul', 'dot', 'cross',
        'basis_change', 'coord_transform', 'std_to_basis', 'basis_to_std',
        'find_matrix', '1', '2', 'both', 'complexity', 'length', 'none',
        'false', 'true', 'rref',
    }
    operation_values = values - non_ops
    assert operation_values <= set(SAMPLES), sorted(operation_values - set(SAMPLES))
    assert {
        'auto', 'expr', 'calculus', 'diff', 'integral', 'limit', 'operations',
        'matrix_symbolic', 'invert', 'ref', 'transform-1', 'transform-2', 'det',
        'dependence', 'projection', 'rank', 'LU', 'orthogonal', 'diag', 'eigen',
        'singular', 'svd', 'schur', 'linear-system', 'linear-solver', 'tangent',
        'linearization', 'critical_points', 'newton', 'arc_length', 'volume_disk',
        'surface_area', 'volume_shell', 'iterated_integral', 'implicit_diff',
        'parametric_diff', 'polar_area', 'polar_slope', 'taylor',
        'series_convergence', 'improper_integral', 'ode', 'numerical_integration',
        'mean_value', 'radius_of_convergence', 'power_series', 'taylor_error',
        'complex_simplify', 'complex_polar', 'complex_roots', 'de_moivre',
    } <= operation_values


def test_every_auto_alias_builds_known_operation():
    dummy = {
        'expression': 'x', 'variable': 'x', 'variable2': 'y', 'variable3': 'z',
        'point': '0', 'point2': '1', 'direction': '+', 'order': '1',
        'lower': '0', 'upper': '1', 'lower2': '0', 'upper2': '1',
        'lower3': '0', 'upper3': '1', 'x1': '1', 'target': '1', 'yvalue': '1',
        'steps': '2', 'curve': 'x', 'surface': 'x+y', 'type': 'ref',
        'target_form': 'rref', 'operations': '+',
    }
    for name, (op, fields) in _AUTO_FUNCS.items():
        assert op in SAMPLES, f'{name} -> {op} has no sample'
        args = [dummy.get(field, 'x') for field in fields]
        data = _build_op_data(name, args)
        assert data['operation_type'] == op
        assert isinstance(data.get('expression', ''), str)


@pytest.mark.parametrize('op', sorted(SAMPLES))
def test_operation_happy_path_latex_contract(op):
    ok, result = start_compute(op, {'operation_type': op, **SAMPLES[op]})
    assert ok, (op, result)
    assert_latex_contract(result)


@pytest.mark.parametrize('op', sorted(SAMPLES))
@pytest.mark.parametrize('bad', MUST_REJECT)
def test_operation_rejects_empty_and_unfilled(op, bad):
    payload = dict(SAMPLES[op])
    payload['expression'] = bad
    ok, result = start_compute(op, {'operation_type': op, **payload})
    assert not ok, (op, bad, result)
    assert isinstance(result, str) and result.strip()
    assert 'Traceback' not in result
    for internal in (
        'MutableDenseMatrix', 'Sympify of expression', 'unhashable type',
        'object has no attribute', 'not enough values', 'Data type not understood',
        'list index out of range',
    ):
        assert internal not in result, (op, bad, result)


@pytest.mark.parametrize('op', sorted(SAMPLES))
@pytest.mark.parametrize('bad', MALFORMED)
def test_operation_rejects_or_safely_handles_malformed_input(op, bad):
    payload = dict(SAMPLES[op])
    payload['expression'] = bad
    ok, result = start_compute(op, {'operation_type': op, **payload})
    if ok:
        assert_latex_contract(result)
    else:
        assert isinstance(result, str) and result.strip()
        assert 'Traceback' not in result
        assert 'list index out of range' not in result
        assert 'Unsupported operation type' not in result


@pytest.mark.parametrize('op', sorted(_MATRIX_LATEX_OPS))
def test_matrix_operation_rejects_ragged_matrix(op):
    # Every MathLive-LaTeX matrix op must reject a ragged matrix.  Parametrize
    # from the registry itself (27 ops) so the 4 previously-missing ops
    # linear-system / transform-1 / transform-2 / matrix_symbolic stay covered.
    assert op in SAMPLES, op
    payload = dict(SAMPLES[op])
    payload['expression'] = '[[1,2],[3]]'
    ok, result = start_compute(op, {'operation_type': op, **payload})
    assert not ok, (op, result)


@pytest.mark.parametrize('op', sorted({
    'invert', 'det', 'eigen', 'schur', 'LU', 'diag', 'trace', 'jordan',
    'definiteness', 'charpoly',
}))
def test_square_only_operation_rejects_non_square(op):
    payload = dict(SAMPLES.get(op, {'expression': '[[1,2,3],[4,5,6]]'}))
    payload['expression'] = '[[1,2,3],[4,5,6]]'
    ok, result = start_compute(op, {'operation_type': op, **payload})
    assert not ok, (op, result)
    assert '方阵' in result


def test_eigen_and_svd_failures_are_not_reported_as_success():
    """A failed decomposition must return success=False, not a step note."""
    for op in ('eigen', 'svd'):
        for matrix in ('[[oo,1],[1,1]]', '[[nan,1],[1,1]]'):
            ok, result = start_compute(op, {
                'operation_type': op, 'expression': matrix})
            assert not ok, (op, matrix, result)
            assert '失败' in result, (op, matrix, result)


def test_linear_transform_failure_is_not_reported_as_success():
    """A singular basis must fail the request, not embed "计算失败" in a success."""
    for payload in (
        {'expression': '[[1,0],[0,1]]\n[[1,1],[1,1]]', 'type': 'find_matrix'},
        {'expression': '[[1,0],[0,1]]\n[[1,0],[0,1]]\n[[1,1],[1,1]]',
         'type': 'change_basis'},
    ):
        ok, result = start_compute('transform-2', {
            'operation_type': 'transform-2', **payload})
        assert not ok, (payload, result)
        assert '失败' in result or '不可逆' in result, result


def test_linear_system_rejects_nonlinear_equations():
    """A nonlinear equation must not be silently turned into a linear system.

    x**2 + y = 2 used to be accepted (the x**2 remainder was swept into b), and
    x*y = 2 was treated as linear with coefficient y.
    """
    for expr in ('x^2+y=2\nx-y=0', 'x*y=2\nx-y=0', 'sin(x)+y=2\nx-y=0'):
        ok, result = start_compute('linear-system', {
            'operation_type': 'linear-system', 'expression': expr})
        assert not ok, (expr, result)
        assert '线性' in result, (expr, result)
    ok, result = start_compute('linear-system', {
        'operation_type': 'linear-system', 'expression': 'x+y=2\nx-y=0'})
    assert ok, result
