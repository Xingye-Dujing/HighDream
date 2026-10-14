# -*- coding: utf-8 -*-
"""Tests for the method-tree -> single-linear-path replay (routes/method_tree_service).

Reuses the same rule engine as the tree build (ManualStepSolver.apply_rule), so a
picked branch reproduces the notebook-identical align block; no full re-search.
"""

from routes.method_tree_service import linear_path_method_tree


def test_replay_filters_synthetic_back_subs():
    # 'back_subs' is a tree-invented node, not a registry rule; it must be dropped from
    # the requested count (finish() performs the substitution automatically).
    res = linear_path_method_tree({
        'domain': 'integral', 'expression': '2*x/(x**2+1)', 'variable': 'x',
        'rule_path': ['substitution', 'logarithmic', 'back_subs'],
    })
    assert res['success'] is True
    assert res['requested'] == 2
    assert res['applied'] == 2
    assert res['done'] is True
    assert res['result'].startswith('\\begin{align}')
    assert len(res['steps']) >= res['applied'] + 1  # >= initial step + applied rules (plus auto +C / back-sub)


def test_replay_bogus_rule_degrades_gracefully():
    # A rule the engine rejects must not raise; it just does not apply.
    res = linear_path_method_tree({
        'domain': 'integral', 'expression': '2*x/(x**2+1)', 'variable': 'x',
        'rule_path': ['not_a_real_rule'],
    })
    assert res['success'] is True
    assert res['applied'] == 0


def test_replay_rejects_non_list_path():
    res = linear_path_method_tree({'domain': 'diff', 'expression': 'x', 'rule_path': 'oops'})
    assert res['success'] is False
