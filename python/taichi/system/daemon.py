from flask import Flask, render_template, send_from_directory, send_file, request
import time
import os
from taichi import get_output_directory, clear_directory_with_suffix
import taichi.tools.video
from flask_cors import CORS
import requests
import json
import base64
import threading

import socket

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

master = False
port = 9563
heart_beat_interval = 1.0

def post(action, data=None):
  return requests.post(url='http://{}:{}/{}'.format(get_master_address(), port, action), json=data)

def get_master_address():
  key = 'TC_MASTER'
  assert key in os.environ, 'master server not found. Please specify master in environment variable {}'.format(key)
  return os.environ[key]

@app.route('/viewer/<path:path>')
def send_front_end_file(path):
  return send_from_directory('viewer', path)

@app.route('/data', methods=['POST'])
def next_frame():
  content = request.get_json(silent=True)
  print(content)
  directory = os.path.join(get_output_directory(), content['path'])
  files = sorted(os.listdir(directory))
  files = list(filter(lambda x: x.endswith('.json'), files))
  frame_fn = '%04d.json' % content['frame_id']
  next_frame = files[(files.index(frame_fn) + content['inc']) % len(files)].split('.')[0]
  next_frame = int(next_frame)

  json_path = os.path.join(directory, frame_fn)
  response = {
    'next_frame': next_frame
  }
  if content['need_geometry']:
    with open(json_path) as f:
      response['data'] = json.loads(f.read())
  return json.dumps(response)

class Server:
  # When ip_address = None, create the local server instance
  def __init__(self, content=None):
    if content is None:
      from taichi.core.util import get_projects
      self.name = socket.gethostname()
      self.ip = post('get_ip').content.decode("utf-8")
      # TODO: get module versions via git
      self.packages = get_projects()
    else:
      self.ip = content['ip']
      self.name = content['name']
      self.packages = content['packages']
      # load from content


  def get_heart_beat(self):
    content = {
      'name': self.name,
      'ip': self.ip,
      'packages': self.packages
    }
    return content

class ServerList:
  def __init__(self):
    self.servers = {}

  def update_srever(self, content):
    ip = content['ip']
    self.servers[ip] = Server(content=content)

@app.route('/get_host_name', methods=['GET'])
def get_hostname():
  return socket.gethostname()

@app.route('/register/<frame_id>', methods=['GET'])
def get_identical(frame_id):
  return str(frame_id)

@app.route('/get_ip', methods=['POST'])
def get_ip():
  return request.remote_addr

class SlaveDaemon:
  def __init__(self):
    server = Server()
    while True:
      hb = server.get_heart_beat()
      print('sending hear beat', hb)
      post(action='heart_beat', data=hb)
      time.sleep(heart_beat_interval)

   #self.th = threading.Thread(target=self.dae)

class MasterDaemon:
  def __init__(self):
    servers = ServerList()
    @app.route('/heart_beat', methods=['POST'])
    def heart_beat():
      content = request.get_json(silent=True)
      servers.update_srever(content)
      return ''
    while True:
      print(servers.servers)
      time.sleep(heart_beat_interval)


def start(master=False):
  if master:
    th = threading.Thread(target=lambda: app.run(port=port, host='0.0.0.0'))
    th.start()
    daemon = MasterDaemon()
  else:
    server = Server()
    daemon = SlaveDaemon()

if __name__ == '__main__':
  start(master=True)

