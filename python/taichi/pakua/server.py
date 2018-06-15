from flask import Flask, render_template, send_from_directory, send_file, request
import os
from taichi import get_output_directory, clear_directory_with_suffix
import taichi.tools.video
from flask_cors import CORS
import requests
import json
import base64

import socket

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

port = 9563

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
    # self.name = requests.get('http://{}:{}/get_host_name'.format(ip_address, port))

    if content is None:
      from taichi.core.util import get_projects
      self.name = socket.gethostname()
      self.ip = socket.gethostname(self.name)
      # TODO: get module versions via git
      self.packages = get_projects()
    else:
      self.ip = content['ip']
      self.name = content['name']
      self.packages = content['packages']
      # load from content


  def get_heart_beat(self):
    content = {
      'ip': self.ip
    }
    return content

class ServerList:
  def __init__(self):
    self.servers = {}

  def update_srever(self, content):
    ip = content['ip']
    self.servers[ip] = Server(content=content)


server = Server()
servers = ServerList()

@app.route('/heart_beat', methods=['POST'])
def heart_beat():
  content = request.get_json(silent=True)
  servers.update_srever(content)
  return ''

@app.route('/get_host_name', methods=['GET'])
def get_hostname():
  return socket.gethostname()

@app.route('/register/<frame_id>', methods=['GET'])
def get_identical(frame_id):
  return str(frame_id)


app.run(port=9563, host='0.0.0.0')

