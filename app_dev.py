from flask import Flask
from config import TEMPLATE_FOLDER, STATIC_FOLDER
from routes import main, api


def create_app():
    """Create and configure the Flask application instance."""
    _app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
                 static_folder=STATIC_FOLDER)
    _app.register_blueprint(main)
    _app.register_blueprint(api, url_prefix='/api')
    return _app


app = create_app()
