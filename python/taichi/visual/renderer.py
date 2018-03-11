from taichi.misc.util import *

import atexit
import os
import time
import numpy

from concurrent.futures import ThreadPoolExecutor
from subprocess import Popen
from tempfile import mkstemp

import taichi

from taichi.core import tc_core
from taichi.misc.util import get_unique_task_id
from taichi.visual.post_process import LDRDisplay
from taichi.misc.settings import get_num_cores
from taichi.gui.image_viewer import show_image


class Renderer(object):

  def __init__(self,
               name=None,
               output_dir=get_unique_task_id(),
               overwrite=True,
               frame=0,
               scene=None,
               preset='pt',
               visualize=True,
               **kwargs):
    self.renderer_name = name
    if output_dir is not None:
      self.output_dir = taichi.settings.get_output_path(output_dir + '/')
    self.post_processor = LDRDisplay()
    self.frame = frame
    self.viewer_started = False
    self.viewer_process = None
    try:
      os.mkdir(self.output_dir)
    except Exception as e:
      if not overwrite:
        print(e)
        exit(-1)
    if scene:
      self.initialize(preset, scene=scene, **kwargs)
    self.visualize = visualize

  def initialize(self, preset=None, scene=None, **kwargs):
    if preset is not None:
      args = Renderer.presets[preset]
      for key, value in list(kwargs.items()):
        args[key] = value
      self.renderer_name = args['name']
    else:
      args = kwargs
    self.c = tc_core.create_renderer(self.renderer_name)
    if scene is not None:
      self.set_scene(scene)
    self.c.initialize(config_from_dict(args))


  def render(self, stages=1000, cache_interval=-1):
    for i in range(1, stages + 1):
      print('stage', i)
      t = time.time()
      self.render_stage()
      print('time:', time.time() - t)
      self.show()
      if cache_interval > 0 and i % cache_interval == 0:
        self.write('img%04d-%06d.png' % (self.frame, i))

    self.write('img%04d-%06d.png' % (self.frame, stages))

  def get_full_fn(self, fn):
    return self.output_dir + fn

  def write(self, fn):
    self.get_image_output().write(self.get_full_fn(fn))

  # Returns numpy.ndarray
  def get_output(self):
    output = self.c.get_output()
    output = image_buffer_to_ndarray(output)

    if self.post_processor:
      output = self.post_processor.process(output)

    return output

  # Returns ImageBuffer<Vector3> a.k.a. Array2DVector3
  def get_image_output(self):
    return taichi.util.ndarray_to_array2d(self.get_output())

  def show(self):
    if not self.visualize:
      return

    show_image('Taichi Renderer', self.get_output())

  def end_viewer_process(self):
    if self.viewer_process.returncode is not None:
      return

    self.viewer_process.terminate()

  def start_viewer(self, frame_path):
    path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), '../gui/tk/viewer.py')

    self.viewer_process = Popen(['python', path, frame_path])

    atexit.register(self.end_viewer_process)

  def __getattr__(self, key):
    return self.c.__getattribute__(key)

  def set_scene(self, scene):
    self.c.set_scene(scene.c)

  def set_post_processor(self, post_processor):
    self.post_processor = post_processor

  presets = {
      'sppm': {
          'name': 'sppm',
          'min_path_length': 1,
          'max_path_length': 10,
          'initial_radius': 0.5,
          'sampler': 'sobol',
          'shrinking_radius': True,
          'num_threads': get_num_cores()
      },
      'vcm': {
          'name': 'vcm',
          'min_path_length': 1,
          'max_path_length': 10,
          'initial_radius': 0.5,
          'sampler': 'prand',
          'stage_frequency': 10,
          'shrinking_radius': True,
          'num_threads': get_num_cores()
      },
      'pt': {
          'name': 'pt',
          'min_path_length': 1,
          'max_path_length': 10,
          'initial_radius': 0.5,
          'sampler': 'sobol',
          'russian_roulette': True,
          'direct_lighting': 1,
          'direct_lighting_light': 1,
          'direct_lighting_bsdf': 1,
          'envmap_is': 1,
          'num_threads': get_num_cores()
      },
      'pt_sdf': {
          'name': 'pt_sdf',
          'min_path_length': 1,
          'max_path_length': 10,
          'initial_radius': 0.5,
          'sampler': 'sobol',
          'russian_roulette': True,
          'direct_lighting': 1,
          'direct_lighting_light': 1,
          'direct_lighting_bsdf': 1,
          'envmap_is': 1,
          'num_threads': get_num_cores()
      },
      'bdpt': {
          'name': 'bdpt',
          'min_path_length': 1,
          'max_path_length': 10,
          'stage_frequence': 3,
          'sampler': 'sobol',
          'num_threads': get_num_cores()
      }
  }
