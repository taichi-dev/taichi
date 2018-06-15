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

master_port = 9563
slave_port = 1116

heart_beat_interval = 1.0

def post_to_master(action, data=None):
  return requests.post(url='http://{}:{}/{}'.format(get_master_address(), master_port, action), json=data)

def get_master_address():
  key = 'TC_MASTER'
  assert key in os.environ, 'master server not found. Please specify master in environment variable {}'.format(key)
  return os.environ[key]

class Server:
  def __init__(self, content=None):
    if content is None:
      from taichi.core.util import get_projects
      self.name = socket.gethostname()
      self.ip = post_to_master('get_ip').content.decode("utf-8")
      # TODO: get module versions via git
      self.packages = get_projects()
    else:
      self.ip = content['ip']
      self.name = content['name']
      self.packages = content['packages']

  def get_heart_beat(self):
    content = {
      'name': self.name,
      'ip': self.ip,
      'packages': self.packages
    }
    return content

  def post(self, action, data=None):
    return requests.post(url='http://{}:{}/{}'.format(self.ip, master_port, action), json=data)

  def call(self, name, data):
    return self.post(action='run_{}'.format(name), data=data)


class ServerList:
  def __init__(self):
    self.servers = {}

  def update_srever(self, content):
    ip = content['ip']
    self.servers[ip] = Server(content=content)

class Job:
  def __init__(self, name, func):
    self.name = name
    self.func = func

class SlaveDaemon:
  def __init__(self):
    server = Server()
    @app.route('/get_host_name', methods=['GET'])
    def get_hostname():
      return socket.gethostname()
    while True:
      hb = server.get_heart_beat()
      print('sending hear beat', hb)
      post_to_master(action='heart_beat', data=hb)
      time.sleep(heart_beat_interval)

class MasterDaemon:
  def __init__(self):
    servers = ServerList()
    @app.route('/heart_beat', methods=['POST'])

    def heart_beat():
      content = request.get_json(silent=True)
      servers.update_srever(content)
      return 'success'

    @app.route('/get_ip', methods=['POST'])
    def get_ip():
      return request.remote_addr

    self.register_job(Job('add', lambda x: {'c': x['a'] + x['b']}))

    while True:
      print('servers', servers.servers)
      time.sleep(heart_beat_interval)
      for name, s in servers.servers.items():
        print(name, s.call('add', {'a':1, 'b':2}).json())

  def register_job(self, job):
    @app.route('/run_{}'.format(job.name), methods=['POST'])
    def run_job():
      return json.dumps(job.func(request.get_json(silent=True)))

def start(master=False):
  if master:
    th = threading.Thread(target=lambda: app.run(port=master_port, host='0.0.0.0'))
    th.start()
    daemon = MasterDaemon()
  else:
    th = threading.Thread(target=lambda: app.run(port=slave_port, host='0.0.0.0'))
    th.start()
    server = Server()
    daemon = SlaveDaemon()

if __name__ == '__main__':
  start(master=True)

