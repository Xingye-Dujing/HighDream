"""Deterministic tests for domains.expression_parser (product-to-sum + parse BFS)."""

from sympy import Symbol, cos, sin, srepr, simplify

from domains.expression_parser import ExpressionParser

x = Symbol('x')
y = Symbol('y')


def _parser():
    return ExpressionParser(max_depth=2)


def test_product_to_sum_identities():
    parser = _parser()
    for prod in (sin(x) * cos(y), cos(x) * cos(y), sin(x) * sin(y)):
        out = parser._product_to_sum_transform(prod)
        assert len(out) == 1
        assert simplify(out[0] - prod) == 0


def test_product_to_sum_skips_non_products():
    parser = _parser()
    assert parser._product_to_sum_transform(sin(x) + cos(y)) == []
    assert parser._product_to_sum_transform(x * y) == []


def test_parse_returns_equivalent_derivations():
    results = _parser().parse('sin(x)*cos(x)')
    assert results
    for expr_obj, reason in results:
        assert isinstance(reason, str)
        assert simplify(expr_obj - sin(x) * cos(x)) == 0


def test_parse_dedupes_by_canonical_key():
    results = _parser().parse('sin(x)*cos(x)')
    keys = [srepr(expr_obj) for expr_obj, _ in results]
    assert len(keys) == len(set(keys))
