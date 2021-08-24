import http.server
import socketserver
from threading import Thread
from webbrowser import open as open_url

from srv.api import run as run_api

def run_interface(host="", port=80):
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer((host, port), Handler)
    print("Serving interface at port", port)
    httpd.serve_forever()

if __name__ == "__main__":
    interface_thread = Thread(target=run_interface, args=("0.0.0.0", 8080))
    interface_thread.setDaemon(True)
    interface_thread.start()
    open_url("http://localhost:8080")
    run_api()