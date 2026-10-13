"""Guard the SymPy assumptions the codebase relies on (documented in AGENTS.md)."""

from sympy import E, Symbol, exp, sympify


def test_implicit_multiplication_is_a_single_symbol():
    # 'ab' must parse as ONE symbol, never a*b
    expr = sympify('ab')
    assert expr.is_Symbol
    assert str(expr) == 'ab'


def test_natural_exponential_is_exp_not_e():
    # bare 'e' is just a symbol; Euler's number is E
    e_sym = sympify('e')
    assert e_sym.is_Symbol
    assert e_sym != E
    # exp(x) is not e**x when 'e' is a plain symbol
    x = Symbol('x')
    assert exp(x) != e_sym ** x
