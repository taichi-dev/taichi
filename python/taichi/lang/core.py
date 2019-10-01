import os
import shutil
from taichi.core.util import get_build_directory
try:
  os.symlink(get_build_directory() + '/libtaichi_lang_core.so', get_build_directory() + '/taichi_lang_core.so')
except:
  pass
try:
  shutil.copy(get_build_directory() + '/libtaichi_lang_core.dylib', get_build_directory() + '/taichi_lang_core.so')
except:
  pass
import taichi_lang_core
