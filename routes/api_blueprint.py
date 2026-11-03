"""The single API blueprint shared by all route modules.

Submodules register their routes on this object at import time; ``routes.api``
is only a small aggregator, so the app factory keeps one registration point.
"""

from flask import Blueprint

api = Blueprint('api', __name__)
