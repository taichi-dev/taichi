import taichi as tc
import taichi.util
# TODO: Remove cv2
import cv2
import time
import os
from taichi.core import tc_core
from taichi.util import *

class Renderer(object):
    def __init__(self, name, output_dir=taichi.util.get_uuid(), overwrite=False):
        self.c = tc_core.create_renderer(name)
        self.output_dir = output_dir + '/'
        self.post_processor = None
        try:
            os.mkdir(self.output_dir)
        except Exception as e:
            if not overwrite:
                print e
                exit(-1)

    def initialize(self, **kwargs):
        self.c.initialize(config_from_dict(kwargs))

    def render(self, stages, cache_interval=1000):
        for i in range(stages):
            print 'stage', i,
            t = time.time()
            self.render_stage()
            print 'time:', time.time() - t
            self.show()
            if i % cache_interval == 0:
                self.write('%07d.png' % i)

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
