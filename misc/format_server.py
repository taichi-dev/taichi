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

  def exec(self, cmd):
    self.writeln(f">>> {cmd}")
    p = subprocess.getoutput(cmd)
    for l in p.split('\n'):
      self.writeln(l)

    return p

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
    self.writeln(f"head commit id {sha}")
    num_commits = int(ret["commits"])
    self.writeln(f"#commits id {num_commits}")

    commits = self.exec(f'git log -n {num_commits + 1} --format="%H"').split(
        '\n')
    fork_commit = commits[num_commits - 1]
    user_id = ret['user']['login']
    branch_name = head['ref']
    ssh_url = head['repo']['ssh_url']
    self.exec(f'git remote add {user_id} {ssh_url}')
    self.exec(f'git fetch {user_id} {branch_name}')
    self.exec(f'git branch -d {user_id}-{branch_name}')
    self.exec(
        f'git checkout -b {user_id}-{branch_name} {user_id}/{branch_name}')
    self.exec(f'ti format {fork_commit}')
    self.exec('git add --all')
    self.exec(f'git commit -m "[skip ci] enforce code format"')
    self.exec(f'git push {user_id} {user_id}-{branch_name}:{branch_name}')

    # self.exec(f'git checkout master')
    def x():
      a =    1


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
