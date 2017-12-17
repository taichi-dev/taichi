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
          'sky', height=0.005 * i + 0.5, direction=0.01 * i)
      envmap_texture.show(res=(512, 512), post_processor=LDRDisplay())
