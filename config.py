import os

# Path settings
TEMPLATE_FOLDER = 'templates'
STATIC_FOLDER = 'static'
TREES_DIR = os.path.join(STATIC_FOLDER, "trees")
os.makedirs(TREES_DIR, exist_ok=True)

# Input render settings
RENDER_SINGLE_MATRIX = ['matrix', 'ref', 'diag', 'eigen', 'rank', 'det', 'dependence',
                        'invert', 'LU', 'svd', 'singular', 'schur', 'orthogonal']
RENDER_MANY_ROWS_ONLY_NUMBER_MATRIX = ['operations', 'linear-solver',
                                       'transform-1', 'transform-2', 'projection']

# Calculator settings
DEFAULT_PARSER_MAX_DEPTH = 3
DEFAULT_LHOPITAL_MAX_COUNT = 5
