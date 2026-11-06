# Path settings
TEMPLATE_FOLDER = 'templates'
STATIC_FOLDER = 'static'

# Input render settings
RENDER_SINGLE_MATRIX = ['matrix', 'ref', 'diag', 'eigen', 'rank', 'det', 'dependence',
                        'invert', 'LU', 'svd', 'singular', 'schur', 'orthogonal']
RENDER_MANY_ROWS_ONLY_NUMBER_MATRIX = ['operations', 'linear-solver',
                                       'transform-1', 'transform-2', 'projection']

# Calculator settings
DEFAULT_PARSER_MAX_DEPTH = 3
DEFAULT_LHOPITAL_MAX_COUNT = 5
# Hard upper limits for user-chosen workload knobs. The UI prefills the defaults
# above and clamps client-side, but API clients can ask for more; without server-
# side caps a single request can monopolize a worker (e.g. order=10**9).
HARD_MAX_DIFF_ORDER = 20
HARD_MAX_PARSER_MAX_DEPTH = 12
HARD_MAX_LHOPITAL_COUNT = 50
# Iteration/step count for applied numerical methods (Newton, bisection, Euler,
# series order, ...). The UI uses small values; without a server-side cap an API
# client could ask for 10**9 steps and pin a worker thread for minutes.
HARD_MAX_ITERATIONS = 200
# Integer-exponent literals with at least this many digits (e.g. 2**1000000 ->
# exponent "1000000" is 7 digits) are parsed with evaluation disabled, so huge
# integer powers are never materialized -- doing so wastes CPU/memory and trips
# Python's int<->str digit-conversion limit when the result is rendered.
LARGE_POWER_EXPONENT_DIGITS = 5

# Task manager settings
MAX_WORKERS = 10
SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS = 120
# Note: This value should be greater than SINGLE_TASK_EXECUTE_TIMEOUT_SECONDS
TASK_CLEAR_AFTER_CREAT_SECONDS = 125

# Method-tree (exhaustive rule enumeration) cutoffs.
# Defaults are what the UI prefills; HARD limits clamp whatever the user asks.
METHOD_TREE_DEFAULT_MAX_DEPTH = 8
METHOD_TREE_DEFAULT_MAX_NODES = 500
METHOD_TREE_DEFAULT_TIME_SECONDS = 30
METHOD_TREE_HARD_MAX_DEPTH = 20
METHOD_TREE_HARD_MAX_NODES = 5000
METHOD_TREE_HARD_MAX_TIME_SECONDS = 600
# Task-manager side timeout for method_tree op_type (must be >= hard time cap
# plus a small grace period so the enumerator's own wall-clock cutoff wins).
METHOD_TREE_TASK_TIMEOUT_SECONDS = METHOD_TREE_HARD_MAX_TIME_SECONDS + 10
# Interactive (step-by-step) mode: effectively no time limit, but bounded
# to 2 hours so the worker thread doesn't live forever if the user walks away.
METHOD_TREE_INTERACTIVE_TASK_TIMEOUT_SECONDS = 7200

# Maximum REF/RREF symbolic-case branches. Each case-split can double the queue
# (and every unordered pair is checked with sympy.satisfiable, which is NP-hard);
# without a hard cap a symbolic matrix with many ambiguous pivots can pin a worker
# forever. Excess branches are reported as truncated, never silently mixed up.
REF_MAX_BRANCHES = 128

# Function-plot sampling (SVG curve via /api/plot). The endpoint is synchronous,
# so the grid is clamped hard to bound CPU/memory; y is rejected beyond PLOT_MAX_ABS_Y
# (asymptotes become null holes the client breaks the polyline at).
PLOT_DEFAULT_SAMPLES = 200
PLOT_MAX_SAMPLES = 1200
PLOT_DEFAULT_WINDOW = 10.0        # x in [-W, W] when the cell gives no bounds
PLOT_MAX_ABS_Y = 1e12
PLOT_COORD_DIGITS = 6
