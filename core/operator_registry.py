"""Declarative operation registry for the universal-box dispatcher.

Keeping the tables here (instead of inside the 1,600-line dispatch module) gives
one obvious place to add an operation and makes the dispatch code read as logic
rather than data.
"""

# Operation types that accept MathLive LaTeX matrices (bmatrix/pmatrix).
_MATRIX_LATEX_OPS = {
    'ref', 'operations', 'invert', 'rank', 'LU', 'diag', 'svd', 'singular',
    'eigen', 'schur', 'det', 'orthogonal', 'projection', 'dependence',
    'linear-system', 'linear-solver', 'transform-1', 'transform-2', 'matrix_symbolic',
    'trace', 'nullspace', 'colspace', 'rowspace', 'pinv', 'jordan', 'definiteness', 'charpoly',
}

# Function-style universal-box name -> (operation_type, positional data fields).
_AUTO_FUNCS = {
    # calculus
    'diff': ('diff', ['expression', 'variable', 'order']),
    'derivative': ('diff', ['expression', 'variable', 'order']),
    'd': ('diff', ['expression', 'variable', 'order']),
    'int': ('integral', ['expression', 'variable', 'lower', 'upper']),
    'integrate': ('integral', ['expression', 'variable', 'lower', 'upper']),
    'limit': ('limit', ['expression', 'variable', 'point', 'direction']),
    'sum': ('finite_sum', ['expression', 'variable', 'lower', 'upper']),
    'fsum': ('finite_sum', ['expression', 'variable', 'lower', 'upper']),
    'prod': ('finite_product', ['expression', 'variable', 'lower', 'upper']),
    'product': ('finite_product', ['expression', 'variable', 'lower', 'upper']),
    'series': ('series_convergence', ['expression', 'variable']),
    'converge': ('series_convergence', ['expression', 'variable']),
    'taylor': ('taylor', ['expression', 'variable', 'point', 'order']),
    'maclaurin': ('taylor', ['expression', 'variable', 'point', 'order']),
    'taylorerror': ('taylor_error', ['expression', 'variable', 'point', 'order']),
    'radius': ('radius_of_convergence', ['expression', 'variable']),
    'radiusofconvergence': ('radius_of_convergence', ['expression', 'variable']),
    'powerseries': ('power_series', ['expression', 'variable', 'point']),
    'improper': ('improper_integral', ['expression', 'variable', 'lower', 'upper']),
    'improperintegral': ('improper_integral', ['expression', 'variable', 'lower', 'upper']),
    'iterated': ('iterated_integral', ['expression', 'variable', 'variable2', 'lower', 'upper', 'lower2', 'upper2']),
    'doubleintegral': ('iterated_integral', ['expression', 'variable', 'variable2', 'lower', 'upper', 'lower2', 'upper2']),
    'tripleintegral': ('triple_integral', ['expression', 'variable', 'variable2', 'variable3',
                                           'lower', 'upper', 'lower2', 'upper2', 'lower3', 'upper3']),
    'triple': ('triple_integral', ['expression', 'variable', 'variable2', 'variable3',
                                   'lower', 'upper', 'lower2', 'upper2', 'lower3', 'upper3']),
    'changeorder': ('changeorder', ['expression', 'variable', 'variable2',
                                    'lower', 'upper', 'lower2', 'upper2']),
    'cylindrical': ('cylindrical', ['expression', 'lower', 'upper', 'lower2', 'upper2', 'lower3', 'upper3']),
    'spherical': ('spherical', ['expression', 'lower', 'upper', 'lower2', 'upper2', 'lower3', 'upper3']),
    'changeorderofintegration': ('changeorder', ['expression', 'variable', 'variable2',
                                                  'lower', 'upper', 'lower2', 'upper2']),
    # applications of differentiation / integration
    'tangent': ('tangent', ['expression', 'variable', 'point']),
    'linearization': ('linearization', ['expression', 'variable', 'point']),
    'linearize': ('linearization', ['expression', 'variable', 'point']),
    'critical': ('critical_points', ['expression', 'variable']),
    'extrema': ('critical_points', ['expression', 'variable']),
    'newton': ('newton', ['expression', 'variable', 'point', 'order']),
    'bisection': ('bisection', ['expression', 'variable', 'lower', 'upper', 'order']),
    'secant': ('secant', ['expression', 'variable', 'point', 'x1', 'order']),
    'fixedpoint': ('fixed_point', ['expression', 'variable', 'point', 'order']),
    'iterate': ('fixed_point', ['expression', 'variable', 'point', 'order']),
    'meanvalue': ('mean_value', ['expression', 'variable', 'lower', 'upper']),
    'rolle': ('mean_value', ['expression', 'variable', 'lower', 'upper']),
    'arclength': ('arc_length', ['expression', 'variable', 'lower', 'upper']),
    'volume': ('volume_disk', ['expression', 'variable', 'lower', 'upper']),
    'disk': ('volume_disk', ['expression', 'variable', 'lower', 'upper']),
    'shell': ('volume_shell', ['expression', 'variable', 'lower', 'upper']),
    'surface': ('surface_area', ['expression', 'variable', 'lower', 'upper']),
    'implicit': ('implicit_diff', ['expression', 'variable']),
    'parametric': ('parametric_diff', ['expression', 'variable']),
    'polararea': ('polar_area', ['expression', 'variable', 'lower', 'upper']),
    'polarslope': ('polar_slope', ['expression', 'variable']),
    'grad': ('gradient', ['expression', 'variable']),
    'gradient': ('gradient', ['expression', 'variable']),
    'hessian': ('hessian', ['expression', 'variable', 'variable2']),
    'jacobian': ('jacobian', ['expression', 'variable']),
    'directional': ('directional', ['expression', 'direction']),
    'directionalderivative': ('directional', ['expression', 'direction']),
    'curl': ('curl', ['expression', 'variable']),
    'div': ('divergence', ['expression', 'variable']),
    'divergence': ('divergence', ['expression', 'variable']),
    'lagrange': ('lagrange', ['expression', 'variable']),
    'lagrangemultiplier': ('lagrange', ['expression', 'variable']),
    'lineint': ('line_integral', ['expression', 'curve', 'variable', 'lower', 'upper']),
    'lineintegral': ('line_integral', ['expression', 'curve', 'variable', 'lower', 'upper']),
    'lineint2': ('line_integral2', ['expression', 'curve', 'variable', 'lower', 'upper']),
    'work': ('line_integral2', ['expression', 'curve', 'variable', 'lower', 'upper']),
    'surfint': ('surface_integral', ['expression', 'surface', 'variable', 'variable2',
                                     'lower', 'upper', 'lower2', 'upper2']),
    'surfaceintegral': ('surface_integral', ['expression', 'surface', 'variable', 'variable2',
                                             'lower', 'upper', 'lower2', 'upper2']),
    'flux': ('flux', ['expression', 'surface', 'variable', 'variable2',
                      'lower', 'upper', 'lower2', 'upper2']),
    'ode': ('ode', ['expression', 'variable']),
    'dsolve': ('ode', ['expression', 'variable']),
    'numerical': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    'numericalintegration': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    'simpson': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    'trapezoid': ('numerical_integration', ['expression', 'variable', 'lower', 'upper', 'order']),
    # numerical IVP solvers: euler(rhs, x, x0, y0, x_end, steps)
    'euler': ('euler', ['expression', 'variable', 'point', 'yvalue', 'target', 'steps']),
    'heun': ('heun', ['expression', 'variable', 'point', 'yvalue', 'target', 'steps']),
    'improvedeuler': ('heun', ['expression', 'variable', 'point', 'yvalue', 'target', 'steps']),
    'rk4': ('rk4', ['expression', 'variable', 'point', 'yvalue', 'target', 'steps']),
    'rungekutta': ('rk4', ['expression', 'variable', 'point', 'yvalue', 'target', 'steps']),
    # complex
    'complex': ('complex_simplify', ['expression']),
    'complexsimplify': ('complex_simplify', ['expression']),
    'complexpolar': ('complex_polar', ['expression']),
    'complexroots': ('complex_roots', ['expression', 'variable']),
    'nthroots': ('complex_nth_roots', ['expression', 'order']),
    'complexnthroots': ('complex_nth_roots', ['expression', 'order']),
    'rootsunity': ('complex_nth_roots', ['expression', 'order']),
    'demoivre': ('de_moivre', ['expression', 'order']),
    # linear algebra
    'det': ('det', ['expression']), 'determinant': ('det', ['expression']),
    'rank': ('rank', ['expression']),
    'trace': ('trace', ['expression']),
    'tr': ('trace', ['expression']),
    'pinv': ('pinv', ['expression']), 'pseudoinverse': ('pinv', ['expression']),
    'jordan': ('jordan', ['expression']), 'jordanform': ('jordan', ['expression']),
    'definite': ('definiteness', ['expression']), 'isdefinite': ('definiteness', ['expression']),
    'sylvestar': ('definiteness', ['expression']),
    'charpoly': ('charpoly', ['expression']), 'characteristic': ('charpoly', ['expression']),
    'nullspace': ('nullspace', ['expression']), 'nullity': ('nullspace', ['expression']),
    'colspace': ('colspace', ['expression']), 'columnspace': ('colspace', ['expression']),
    'rowspace': ('rowspace', ['expression']),
    'inverse': ('invert', ['expression']), 'inv': ('invert', ['expression']),
    'transpose': ('operations', ['expression']),
    'rref': ('ref', ['expression']), 'ref': ('ref', ['expression']), 'rowreduce': ('ref', ['expression']),
    'eigen': ('eigen', ['expression']), 'eigenvals': ('eigen', ['expression']),
    'eigenvects': ('eigen', ['expression']),
    'svd': ('svd', ['expression']), 'singular': ('svd', ['expression']),
    'schur': ('schur', ['expression']),
    'lu': ('LU', ['expression']),
    'diag': ('diag', ['expression']), 'diagonalize': ('diag', ['expression']),
    'orthogonal': ('orthogonal', ['expression']), 'orthogonalize': ('orthogonal', ['expression']),
    'gramschmidt': ('orthogonal', ['expression']),
    'dependent': ('dependence', ['expression']), 'dependence': ('dependence', ['expression']),
    'project': ('projection', ['expression']), 'projection': ('projection', ['expression']),
    # symbolic / expression
    'simplify': ('expr', ['expression']), 'expand': ('expr', ['expression']),
    'factor': ('expr', ['expression']), 'equivalent': ('expr', ['expression']),
    'msimplify': ('matrix_symbolic', ['expression']),
    'matrixsimplify': ('matrix_symbolic', ['expression']),
    'symmatrix': ('matrix_symbolic', ['expression']),
    # matrix arithmetic via the universal box (two matrix args joined by newline)
    'add': ('operations', []), 'subtract': ('operations', []), 'minus': ('operations', []),
    'multiply': ('operations', []), 'matmul': ('operations', []),
}

# op -> fixed extra data fields the calculators require.
_AUTO_DEFAULTS = {
    'det': {'type': '1'}, 'ref': {'target_form': 'rref'}, 'svd': {},
    'expr': {'sort_strategy': 'simplify'},
}

# Ops whose calculators read a single ``expression`` holding several matrices /
# vectors separated by newlines: every function argument is joined with '\n'.
_AUTO_MULTILINE_OPS = {'operations', 'projection', 'dependence', 'orthogonal'}

# Function name -> fixed data fields that the name itself decides.
_AUTO_NAME_EXTRAS = {
    'add': {'operations': '+'}, 'subtract': {'operations': '-'}, 'minus': {'operations': '-'},
    'multiply': {'operations': '*'}, 'matmul': {'operations': '*'},
    'transpose': {'operations': 'transpose'},
}

# Matrix-function names that may appear bare (e.g. \det [[1,2],[3,4]]).
_MATRIX_OP_NAMES = {
    'det', 'determinant', 'rank', 'inverse', 'inv', 'rref', 'ref', 'rowreduce',
    'eigen', 'eigenvals', 'eigenvects', 'svd', 'singular', 'schur', 'lu', 'diag',
    'diagonalize', 'orthogonal', 'orthogonalize', 'gramschmidt', 'dependent',
    'dependence', 'project', 'projection', 'transpose',
    'trace', 'tr', 'nullspace', 'nullity', 'colspace', 'columnspace', 'rowspace',
    'pinv', 'pseudoinverse', 'jordan', 'jordanform',
    'definite', 'isdefinite', 'sylvestar',
    'charpoly', 'characteristic',
}

# Ops whose engine consumes a RESOLVED value, so a nested operator is worth hoisting.
_CALC_COMPOSABLE = {'diff', 'integral', 'limit', 'expr'}

# Ops that only make sense for a square matrix.
_SQUARE_MATRIX_OPS = {
    'invert', 'det', 'eigen', 'schur', 'LU', 'diag', 'trace',
    'jordan', 'definiteness', 'charpoly',
}

# Operations whose expression is a structured text (not a single scalar/LaTeX
# formula), so the generic scalar normalizer must not rewrite it.
_TEXT_INPUT_OPS = {
    'ode', 'linear-system', 'parametric_diff',
    # '|'-separated component/variable fields (vector fields, Jacobians, ...)
    'jacobian', 'curl', 'divergence', 'lagrange', 'line_integral2', 'flux',
}

# ``ode`` / symbolic-matrix expressions legitimately contain operator notation.
_AUTO_ROUTE_SKIP = {'ode', 'matrix_symbolic'}
