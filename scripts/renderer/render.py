import numpy as np
from taichi_utils import *
# TODO: Remove cv2
import cv2
import time
import math

import renderer.assets as assets

class ConfigInitializable:
    def __init__(self, **kwargs):
        self.c = None
        self.c.initialize(config_from_dict(kwargs))


class Renderer(object):
    def __init__(self, name, output_fn):
        self.c = tc.create_renderer(name)
        self.output_fn = output_fn

    def initialize(self, **kwargs):
        self.c.initialize(config_from_dict(kwargs))

    def render(self, stages):
        for i in range(stages):
            print 'stage', i,
            t = time.time()
            self.render_stage()
            print 'time:', time.time() - t
            self.show()

    def show(self):
        self.write_output(self.output_fn)
        cv2.imshow('Rendered', cv2.imread(self.output_fn))
        cv2.waitKey(1)

    def __getattr__(self, key):
        return self.c.__getattribute__(key)

class Camera:
    def __init__(self, name, **kwargs):
        self.c = tc.create_camera(name)
        self.c.initialize(config_from_dict(kwargs))

