import os
import taichi as tc
try:
  os.symlink(tc.get_build_directory() + '/libtaichi_lang.so', tc.get_build_directory() + '/taichi_lang.so')
except:
  pass
import taichi_lang
