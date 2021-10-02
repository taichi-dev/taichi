from taichi.core.settings import get_os_name
from taichi.core.util import *
# TODO: move this to taichi/__init__.py.
#       This is blocked since we currently require importing this before taichi.lang
#       but yapf refuses to give up formatting there.
from taichi.type import *

ti_core.build = build
ti_core.load_module = load_module

__all__ = [s for s in dir() if not s.startswith('_')]
