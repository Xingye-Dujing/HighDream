"""API route aggregator.

Route groups live in focused modules (compute / matrix / sympy / manual /
method tree / snapshot).  Importing this module registers them on the shared
``api`` blueprint, while ``routes/__init__`` continues to expose that blueprint.

Compatibility note: ``_manual_sessions`` is re-exported because the manual
endpoints and tests use it as the in-memory session store.
"""

from .api_blueprint import api
from . import (  # pylint: disable=unused-import
    compute_api,
    manual_api,
    matrix_api,
    method_tree_api,
    snapshot_api,
    sympy_api,
)
from .manual_api import _manual_sessions

__all__ = ['api', '_manual_sessions']
