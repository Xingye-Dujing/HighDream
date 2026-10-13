"""The step engines must render all math as live LaTeX, never literal ``$``."""

from sympy import Symbol

from domains import DiffCalculator, IntegralCalculator, LimitCalculator

x = Symbol('x')


def test_core_outputs_have_no_literal_dollar():
    outputs = [
        DiffCalculator().compute_latex('x**2*sin(x)', x),
        IntegralCalculator().compute_latex('x**2', x),
        LimitCalculator().compute_latex('sin(x)/x', x, 0, 'both'),
    ]
    for out in outputs:
        assert '$' not in out
        assert out.startswith(r'\begin{align}')
