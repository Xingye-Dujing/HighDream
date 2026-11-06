"""Textbook rational integrals must use a direct rule, not a minutes-long BFS.

Regression: ``integral('1/(x^2+2*x+5)')`` missed the arctan rule (a nested-sum
parse artifact, fixed in ``test_parse_canonical.py``) and then integration by
parts spawned ever-harder sub-integrals. The repeated denominator
``1/(x^2+2*x+5)^2`` likewise had no direct rule and could pin a worker for
minutes.
"""

import time

from routes.compute_service import start_compute


def _integral(expression):
    started = time.time()
    ok, res = start_compute('integral', {
        'operation_type': 'integral', 'expression': expression,
        'variable': 'x'})
    assert ok, res
    assert time.time() - started < 20, (expression, 'integral took too long')
    return str(res)


def test_rational_integral_uses_the_direct_rule_quickly():
    res = _integral('1/(x^2+2*x+5)')
    assert 'atan' in res, res


def test_repeated_quadratic_denominator_is_bounded():
    res = _integral('1/(x^2+2*x+5)^2')
    assert 'atan' in res, res
