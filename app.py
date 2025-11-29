# NOTE: This import is required when packaging with PyInstaller !!!
# import matplotlib.backends.backend_svg

from flask import Flask
from config import TEMPLATE_FOLDER, STATIC_FOLDER
from routes import main, api


def create_app():
    """Create and configure the Flask application instance.

    Returns:
        Flask: A configured Flask application instance with registered blueprints.
    """
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
                static_folder=STATIC_FOLDER)
    app.register_blueprint(main)
    app.register_blueprint(api)
    return app


if __name__ == '__main__':
    # Allow access from other devices on the local network
    # It doesn't work when both phone and computer are connected to the campus network;
    # you can share the hotspot from the phone to the computer, then access via the phone's local network
    # Note: Currently, I haven't considered mobile adaptation, it's better not to use mobile devices to access this website
    create_app().run(debug=True, host='0.0.0.0', port=5000)
