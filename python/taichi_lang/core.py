import os
import shutil
import taichi as tc
try:
  os.symlink(tc.get_build_directory() + '/libtaichi_lang_core.so', tc.get_build_directory() + '/taichi_lang_core.so')
except:
  pass
try:
  shutil.copy(tc.get_build_directory() + '/libtaichi_lang_core.dylib', tc.get_build_directory() + '/taichi_lang_core.so')
except:
  pass
import taichi_lang_core
