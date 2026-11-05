"""
For Vercel deployment.
"""

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from config import TEMPLATE_FOLDER, STATIC_FOLDER
from routes import main, api


def create_app():
    """Create and configure the Flask application instance."""
    _app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
                 static_folder=STATIC_FOLDER)
    # Bound every request body so a huge payload cannot be buffered before the
    # per-endpoint checks run (snapshots keep a tighter 2 MB cap inside
    # ``snapshot_store``).
    _app.config['MAX_CONTENT_LENGTH'] = 4 * 1024 * 1024  # 4 MB
    _app.register_blueprint(main)
    _app.register_blueprint(api, url_prefix='/api')

    @_app.errorhandler(RequestEntityTooLarge)
    def _payload_too_large(_error):
        return jsonify({'success': False, 'error': '请求体过大'}), 413

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
