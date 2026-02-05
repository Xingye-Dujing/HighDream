import time
import threading
import webbrowser

# NOTE: This import is required when packaging with PyInstaller !!!
# import matplotlib.backends.backend_svg


from gevent import pywsgi
from flask import Flask
from config import TEMPLATE_FOLDER, STATIC_FOLDER
from routes import main, api

# matplotlib.use('Agg')


def create_app():
    """Create and configure the Flask application instance.

    Returns:
        Flask: A configured Flask application instance with registered blueprints.
    """
    app = Flask(__name__, template_folder=TEMPLATE_FOLDER,
                static_folder=STATIC_FOLDER)
    app.register_blueprint(main)
    app.register_blueprint(api, url_prefix='/api')
    return app


if __name__ == '__main__':
    # Function to run the server
    def start_server():
        server = pywsgi.WSGIServer(('127.0.0.1', 5000), create_app(), log=None)
        server.serve_forever()

    # Start server in a separate thread
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()

    print("\n启动服务...\n")

    # Give the server a moment to start
    time.sleep(0.1)

    # Open the browser
    webbrowser.open_new('http://127.0.0.1:5000/')

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n关闭中...")
