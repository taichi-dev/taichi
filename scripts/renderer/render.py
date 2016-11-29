import numpy as np
from taichi_utils import *
# TODO: Remove cv2
import cv2
import time
import os
import math

import renderer.assets as assets

class ConfigInitializable:
    def __init__(self, **kwargs):
        self.c = None
        self.c.initialize(config_from_dict(kwargs))


class Renderer(object):
    def __init__(self, name, output_dir):
        self.c = tc.create_renderer(name)
        self.output_dir = output_dir + '/'
        os.mkdir(self.output_dir)

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
        self.write_output(self.get_full_fn(fn))

    def show(self):
        self.write('tmp.png')
        cv2.imshow('Rendered', cv2.imread(self.get_full_fn('tmp.png')))
        cv2.waitKey(1)

    def __getattr__(self, key):
        return self.c.__getattribute__(key)

class Camera:
    def __init__(self, name, **kwargs):
        self.c = tc.create_camera(name)
        self.c.initialize(config_from_dict(kwargs))

