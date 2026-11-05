"""Contract tests for the declarative operation registry.

``core/operator_registry.py`` holds the tables that drive auto-routing, matrix
normalization and square-matrix validation; ``routes/compute_service.py`` holds
the calculator-class lookups.  Every entry must stay in sync with the
known-good samples in ``test_operation_contract.SAMPLES`` so a newly added
operation cannot ship without a sample, and every calculator class must stay
constructible.
"""

import pytest

from core import operator_registry as registry
from core.input_validation import looks_like_operator_input, should_route_as_operator
from core.latex_syntax import _iter_op_calls, _normalize_latex_ops
from routes import compute_service as service
from tests.test_operation_contract import ALL_OPS, SAMPLES

# Tables whose members ARE runtime operation_type strings: each must have a
# known-good sample (and be part of the discovered operation universe).
OPERATION_TYPE_TABLES = {
    '_MATRIX_LATEX_OPS': registry._MATRIX_LATEX_OPS,
    '_SQUARE_MATRIX_OPS': registry._SQUARE_MATRIX_OPS,
    '_TEXT_INPUT_OPS': registry._TEXT_INPUT_OPS,
    '_AUTO_DEFAULTS': set(registry._AUTO_DEFAULTS),
    '_AUTO_MULTILINE_OPS': registry._AUTO_MULTILINE_OPS,
    '_CALC_COMPOSABLE': registry._CALC_COMPOSABLE,
    '_AUTO_ROUTE_SKIP': registry._AUTO_ROUTE_SKIP,
}

# Tables whose members are universal-box FUNCTION names, not operation_type
# strings: they do not map 1:1 to SAMPLES.  They must instead be registered
# aliases resolving to an operation that has a sample.
FUNCTION_NAME_TABLES = {
    '_MATRIX_OP_NAMES': registry._MATRIX_OP_NAMES,
    '_AUTO_NAME_EXTRAS': set(registry._AUTO_NAME_EXTRAS),
}


@pytest.mark.parametrize('table,members', sorted(OPERATION_TYPE_TABLES.items()))
def test_operation_type_tables_are_covered_by_samples(table, members):
    assert members, table
    assert members <= set(SAMPLES), (table, sorted(members - set(SAMPLES)))
    assert members <= ALL_OPS, (table, sorted(members - ALL_OPS))


@pytest.mark.parametrize('table,members', sorted(FUNCTION_NAME_TABLES.items()))
def test_function_name_tables_resolve_to_sampled_operations(table, members):
    assert members, table
    unknown = members - set(registry._AUTO_FUNCS)
    assert not unknown, (table, sorted(unknown))
    for name in members:
        op = registry._AUTO_FUNCS[name][0]
        assert op in SAMPLES, (table, name, op)


def test_auto_route_skip_ops_are_excluded_from_operator_rerouting():
    # _AUTO_ROUTE_SKIP is NOT a set of dispatch targets: it lists the ops whose
    # input must be parsed as DATA (ode / symbolic matrices), so their box must
    # never be re-routed through the explicit-operator dispatcher.  Assert that
    # meaning directly, in addition to the sample coverage checked above.
    for op in registry._AUTO_ROUTE_SKIP:
        assert op in SAMPLES, op
        assert should_route_as_operator('diff(x^2,x)', op) is False, op
    # Control: a normal op carrying the same explicit template IS re-routed.
    assert should_route_as_operator('diff(x^2,x)', 'integral') is True


def test_registry_table_sizes_are_pinned():
    # Pin the cardinalities so a table that silently loses entries fails here.
    assert len(registry._MATRIX_LATEX_OPS) == 27
    assert len(registry._MATRIX_OP_NAMES) == 41
    assert len(registry._AUTO_ROUTE_SKIP) == 2


CALCULATOR_ENTRIES = (
    [('_calculators', key, cls) for key, cls in service._calculators.items()]
    + [('_cached_calculators', key, cls) for key, cls in service._cached_calculators.items()]
)


@pytest.mark.parametrize(
    'table,key,calculator', CALCULATOR_ENTRIES,
    ids=[f'{table}:{key}' for table, key, _ in CALCULATOR_ENTRIES],
)
def test_registry_calculator_classes_instantiable(table, key, calculator):
    assert isinstance(key, str) and key, (table, key)
    assert isinstance(calculator, type), (table, key, calculator)
    instance = calculator()  # construction only: no real computation
    assert instance is not None, (table, key)


def test_calculator_lookup_tables_are_not_empty():
    assert service._calculators
    assert service._cached_calculators


@pytest.mark.parametrize('name', sorted(registry._MATRIX_OP_NAMES))
def test_matrix_op_name_is_recognized_by_the_operator_scanner(name):
    # A bare matrix spelling such as ``det [[1,2],[3,4]]`` is folded into an
    # operator call; verify both scanner layers recognize the 41 names and that
    # each name assembles a payload for a sampled operation (no computation).
    assert name in registry._AUTO_FUNCS, name
    expression = f'{name}([[1,2],[3,4]])'
    calls = _iter_op_calls(_normalize_latex_ops(expression))
    assert any(call['name'] == name for call in calls), name
    assert looks_like_operator_input(expression) is True, name
    op = registry._AUTO_FUNCS[name][0]
    assert op in SAMPLES, (name, op)
    data = service._build_op_data(name, ['[[1,2],[3,4]]'])
    assert data['operation_type'] == op, (name, data)
