import argparse
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import subprocess


class TaichiFormatServer(BaseHTTPRequestHandler):

  def _set_headers(self):
    self.send_response(200)
    self.send_header("Content-type", "text/html")
    self.end_headers()

  def _html(self, message):
    content = f"<html><body>{message}</body></html>"
    return content.encode("utf8")

  def writeln(self, f):
    self.wfile.write(self._html(f + '<br>'))

  def do_GET(self):
    print(self.path)
    path = self.path[1:]
    self._set_headers()
    if not path.isdigit():
      self.writeln(
          "Error: Invalid input format. Usage example: https://server:8000/12345, where '12345' is the PR id"
      )
      return
    pr_id = int(path)

    ret = requests.get(
        f'https://api.github.com/repos/taichi-dev/taichi/pulls/{pr_id}')
    if ret.status_code == 404:
      self.writeln(f"Error: PR {pr_id} not found!")
      return
    ret = ret.json()
    url = ret['url']

    self.writeln(
        f"Processing <a href='https://github.com/taichi-dev/taichi/pull/{pr_id}'>PR {pr_id}</a>"
    )
    self.writeln(f"[<a href='{url}'>Metadata</a>]")
    head = ret["head"]
    repo_url = head["repo"]["html_url"]
    sha = head["sha"]
    self.writeln(f"repo url {repo_url}")
    self.writeln(f"commit id {sha}")
    p = subprocess.getoutput("ti format")
    for l in p.split('\n'):
      self.writeln(l)


def run(addr, port):
  server_address = (addr, port)
  httpd = HTTPServer(server_address, TaichiFormatServer)
  print(f"Starting Taichi format server on {addr}:{port}")
  httpd.serve_forever()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Run the Taichi format server")
  parser.add_argument(
      "-l",
      "--listen",
      default="localhost",
      help="Specify the IP address on which the server listens",
  )
  parser.add_argument(
      "-p",
      "--port",
      type=int,
      default=8000,
      help="Port on which the server listens",
  )
  args = parser.parse_args()
  run(addr=args.listen, port=args.port)
