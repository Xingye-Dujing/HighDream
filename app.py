"""
For Vercel deployment.
"""

from flask import Flask, jsonify, request

from config import TEMPLATE_FOLDER, STATIC_FOLDER
from routes import main, api


def create_app():
    """Create and configure the Flask application instance."""
    _app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
                 static_folder=STATIC_FOLDER)
    _app.register_blueprint(main)
    _app.register_blueprint(api, url_prefix='/api')

    @_app.before_request
    def _validate_api_json():
        """Reject non-object JSON bodies before any endpoint calls ``.get``.

        A JSON array (or a missing/invalid content type) used to raise
        ``AttributeError``/415 inside individual handlers.  Page routes may
        still use ordinary form POSTs, so only ``/api/`` writes are checked.
        """
        if request.method not in ('POST', 'PUT', 'PATCH'):
            return None
        if not request.path.startswith('/api/'):
            return None
        error = jsonify({'success': False, 'error': '请求体必须是 JSON 对象'}), 400
        if not request.is_json:
            return error
        if not isinstance(request.get_json(silent=True), dict):
            return error
        return None

    return _app


app = create_app()

if __name__ == '__main__':
    app.run()
