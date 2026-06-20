# import os

# Path settings
TEMPLATE_FOLDER = 'templates'
STATIC_FOLDER = 'static'
TREES_DIR = STATIC_FOLDER + "/trees"
# os.makedirs(TREES_DIR, exist_ok=True)

# Input render settings
RENDER_SINGLE_MATRIX = ['matrix', 'ref', 'diag', 'eigen', 'rank', 'det', 'dependence',
                        'invert', 'LU', 'svd', 'singular', 'schur', 'orthogonal']
RENDER_MANY_ROWS_ONLY_NUMBER_MATRIX = ['operations', 'linear-solver',
                                       'transform-1', 'transform-2', 'projection']

# Calculator settings
DEFAULT_PARSER_MAX_DEPTH = 3
DEFAULT_LHOPITAL_MAX_COUNT = 5

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
METHOD_TREE_HARD_MAX_TIME_SECONDS = 60
# Task-manager side timeout for method_tree op_type (must be >= hard time cap
# plus a small grace period so the enumerator's own wall-clock cutoff wins).
METHOD_TREE_TASK_TIMEOUT_SECONDS = METHOD_TREE_HARD_MAX_TIME_SECONDS + 10
