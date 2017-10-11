from flask import Flask, render_template, send_from_directory, send_file, request
from taichi.misc.settings import get_output_directory
import os
from taichi import get_output_directory
from flask_cors import CORS
import json

app = Flask(__name__)
#CORS(app)


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


@app.route('/')
def browse_outputs():
  output_dir = get_output_directory()
  dirs = os.listdir(output_dir)
  dirs = sorted(dirs)
  entries = []
  for d in dirs:
    entries.append({
        'title': d,
        'text': '',
    })
  return render_template('browser.html', entries=entries)


@app.route('/view/<folder>')
def view(folder):
  output_dir = get_output_directory()
  return render_template('view.html', folder=folder)


def get_pakua_server():
  return app
