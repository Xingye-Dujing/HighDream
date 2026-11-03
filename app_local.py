import threading
import time
import webbrowser

from gevent import pywsgi

from app import create_app


# NOTE: This import is required when packaging with PyInstaller !!!
# import matplotlib.backends.backend_svg


# matplotlib.use('Agg')


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
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n关闭中...")
