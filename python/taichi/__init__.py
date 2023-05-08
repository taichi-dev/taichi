from taichi._funcs import *
from taichi._lib import core as _ti_core
from taichi._lib.utils import warn_restricted_version
from taichi._logging import *
from taichi._snode import *
from taichi.lang import *  # pylint: disable=W0622 # TODO(archibate): It's `taichi.lang.core` overriding `taichi.core`
from taichi.types.annotations import *

# Provide a shortcut to types since they're commonly used.
from taichi.types.primitive_types import *


from taichi import ad, algorithms, experimental, graph, linalg, math, sparse, tools, types
from taichi.ui import GUI, hex_to_rgb, rgb_to_hex, ui

# Issue#2223: Do not reorder, or we're busted with partially initialized module
from taichi import aot  # isort:skip


def __getattr__(attr):
    if attr == "cfg":
        return None if lang.impl.get_runtime().prog is None else lang.impl.current_cfg()
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")


__version__ = (
    _ti_core.get_version_major(),
    _ti_core.get_version_minor(),
    _ti_core.get_version_patch(),
)

del _ti_core

warn_restricted_version()
del warn_restricted_version
