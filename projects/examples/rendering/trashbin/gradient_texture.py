import math
from taichi.misc.util import Vector
from taichi.visual import *
from taichi.visual.post_process import *
from taichi.visual.texture import Texture
import time

if __name__ == '__main__':
  while True:
    for i in range(100):
      envmap_texture = Texture(
          'spherical_gradient',
          inside_val=(1, 1, 10, 1),
          outside_val=(0.5, 0.5, 0.5, 1),
          angle=3,
          sharpness=5)
      envmap_texture.show(res=(512, 512), post_processor=LDRDisplay())
