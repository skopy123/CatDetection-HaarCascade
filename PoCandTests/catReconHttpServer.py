
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 8080

class MyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        # read the content-length header
        content_length = int(self.headers.get("Content-Length"))
        # read that many bytes from the body of the request
        body = self.rfile.read(content_length)

        self.send_response(200)
        self.end_headers()
        # echo the body in the response
        self.wfile.write(body)


def httpServerloop():
  # PORT = 8080
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    print("serving at port", PORT)
    try:
        httpd.server_forever()
    except KeyboardInterrupt:
        print("Going down...")

http_thread = threading.Thread(target=httpServerloop)
http_thread.start()

#cmd line loop
#while True:
#    user_input = input()
#    if user_input == "X":
#        print("x readed")
#        http_thread._stop()
#        break
