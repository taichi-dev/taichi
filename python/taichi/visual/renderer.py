from taichi.misc.util import *
import time
import os
import taichi
from taichi.core import tc_core
from taichi.misc.util import get_uuid
from taichi.visual.post_process import LDRDisplay
from taichi.misc.settings import get_num_cores
import cv2


class Renderer(object):
    def __init__(self, name=None, output_dir=get_uuid(), overwrite=True, frame=0,
                 scene=None, preset=None, **kwargs):
        self.renderer_name = name
        self.output_dir = taichi.settings.get_output_path(output_dir + '/')
        self.post_processor = LDRDisplay()
        self.frame = frame
        try:
            os.mkdir(self.output_dir)
        except Exception as e:
            if not overwrite:
                print e
                exit(-1)
        if scene:
            self.initialize(preset, scene=scene, **kwargs)

    def initialize(self, preset=None, scene=None, **kwargs):
        if preset is not None:
            args = Renderer.presets[preset]
            for key, value in kwargs.items():
                args[key] = value
            self.renderer_name = args['name']
        else:
            args = kwargs
        self.c = tc_core.create_renderer(self.renderer_name)
        if scene is not None:
            self.set_scene(scene)
        self.c.initialize(config_from_dict(args))

    def render(self, stages, cache_interval=-1):
        for i in range(1, stages + 1):
            print 'stage', i
            t = time.time()
            self.render_stage()
            print 'time:', time.time() - t
            self.show()
            if cache_interval > 0 and i % cache_interval == 0:
                self.write('img%04d-%06d.png' % (self.frame, i))

        self.write('img%04d-%06d.png' % (self.frame, stages))

    def get_full_fn(self, fn):
        return self.output_dir + fn

    def write(self, fn):
        cv2.imwrite(self.get_full_fn(fn), self.get_output() * 255)

    def get_output(self):
        output = self.c.get_output()
        output = image_buffer_to_ndarray(output)
        if self.post_processor:
            output = self.post_processor.process(output)
        return output

    def show(self):
        cv2.imshow('Rendered', self.get_output())
        cv2.waitKey(1)

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
        'bdpt': {
            'name': 'bdpt',
            'min_path_length': 1,
            'max_path_length': 10,
            'stage_frequence': 3,
            'sampler': 'sobol',
            'num_threads': get_num_cores()
        }
    }
