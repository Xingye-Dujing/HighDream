# -*- coding: utf-8 -*-
"""Regression tests for single-step undo on the manual /rule_select path.

A single "apply rule" can commit MANY derivation steps (rules write into the step
generator, _advance_pending auto-commits visited sub-expressions, finish() appends
back-substitution / +C). So undo must roll back the whole action via a snapshot and
yield byte-identical state to just before it — not pop one step."""

import pytest

from core.manual_step_solver import ManualStepSolver

CASES = [
    ('diff', 'x**2 + sin(x)', {},),
    ('integral', 'x*exp(x**2)', {}),
    ('limit', 'sin(3*x)/x', {'point': 0}),
]


def _first_rule(solver):
    rules = solver.applicable_rules()
    assert rules, "expected at least one applicable rule"
    return rules[0]['name']


@pytest.mark.parametrize('domain,expression,extra', CASES)
def test_undo_restores_pre_action_state(domain, expression, extra):
    s = ManualStepSolver(domain=domain, expression=expression, variable='x', **extra)
    rule = _first_rule(s)
    before = s.state()
    after = s.apply_rule(rule)
    assert after != before          # the action changed the derivation
    undone = s.undo()
    assert undone == before         # byte-identical restore across all state keys


def test_undo_multi_level_unwinds():
    s = ManualStepSolver(domain='diff', expression='x**2 + sin(x) + 3*x', variable='x')
    history = [s.state()]
    for _ in range(3):
        if s.done or not s.applicable_rules():
            break
        s.apply_rule(s.applicable_rules()[0]['name'])
        history.append(s.state())
    # unwind every applied step; each undo must match the corresponding earlier state
    for k in range(1, len(history)):
        assert s.undo() == history[len(history) - 1 - k]


def test_undo_on_fresh_session_raises():
    s = ManualStepSolver(domain='diff', expression='x**2', variable='x')
    with pytest.raises(RuntimeError):
        s.undo()
