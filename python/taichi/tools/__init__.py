"""Taichi utility module.

- `image` submodule for image io.
- `video` submodule for exporting results to video files.
- `diagnose` submodule for printing system environment information.
"""
from taichi.tools.diagnose import *
from taichi.tools.image import *
from taichi.tools.np2ply import *
from taichi.tools.video import *
from taichi.tools.vtk import *
