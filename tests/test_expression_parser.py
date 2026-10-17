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


def test_product_to_sum_handles_symbolic_and_three_factors():
    parser = _parser()
    # non-numeric coefficient kept as multiplier
    out = parser._product_to_sum_transform(x * sin(x) * cos(x))
    assert len(out) == 1 and simplify(out[0] - x * sin(x) * cos(x)) == 0
    # three trig factors: first pair converted, third kept
    out2 = parser._product_to_sum_transform(sin(x) * cos(x) * sin(y))
    assert len(out2) == 1 and simplify(out2[0] - sin(x) * cos(x) * sin(y)) == 0


def test_new_transforms_are_equivalent():
    parser = _parser()
    from sympy import sqrt
    # collect groups powers of x
    collected = parser._collect_transform(x**3 + x**2 + x*y + x)
    assert len(collected) == 1 and simplify(collected[0] - (x**3 + x**2 + x*y + x)) == 0
    # sqrtdenest denests a nested radical
    denested = parser._sqrtdenest_transform(sqrt(2 + sqrt(3)))
    assert len(denested) == 1 and simplify(denested[0]**2 - (2 + sqrt(3))) == 0
    # ratsimp combines a sum of fractions
    combined = parser._ratsimp_transform(1/x + 1/y)
    assert len(combined) == 1 and simplify(combined[0] - (1/x + 1/y)) == 0


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
