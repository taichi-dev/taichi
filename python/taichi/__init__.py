import sys

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

__deprecated_names__ = {
    "SOA": "Layout.SOA",
    "AOS": "Layout.AOS",
    "print_profile_info": "profiler.print_scoped_profiler_info",
    "clear_profile_info": "profiler.clear_scoped_profiler_info",
    "print_memory_profile_info": "profiler.print_memory_profiler_info",
    "CuptiMetric": "profiler.CuptiMetric",
    "get_predefined_cupti_metrics": "profiler.get_predefined_cupti_metrics",
    "print_kernel_profile_info": "profiler.print_kernel_profiler_info",
    "query_kernel_profile_info": "profiler.query_kernel_profiler_info",
    "clear_kernel_profile_info": "profiler.clear_kernel_profiler_info",
    "kernel_profiler_total_time": "profiler.get_kernel_profiler_total_time",
    "set_kernel_profiler_toolkit": "profiler.set_kernel_profiler_toolkit",
    "set_kernel_profile_metrics": "profiler.set_kernel_profiler_metrics",
    "collect_kernel_profile_metrics": "profiler.collect_kernel_profiler_metrics",
    "VideoManager": "tools.VideoManager",
    "PLYWriter": "tools.PLYWriter",
    "imread": "tools.imread",
    "imresize": "tools.imresize",
    "imshow": "tools.imshow",
    "imwrite": "tools.imwrite",
    "ext_arr": "types.ndarray",
    "any_arr": "types.ndarray",
    "Tape": "ad.Tape",
    "clear_all_gradients": "ad.clear_all_gradients",
}

__customized_deprecations__ = {
    "parallelize": (
        "Please use ti.loop_config(parallelize=...) instead.",
        "lang.misc._parallelize",
    ),
    "serialize": (
        "Please use ti.loop_config(serialize=True) instead.",
        "lang.misc._serialize",
    ),
    "block_dim": (
        "Please use ti.loop_config(block_dim=...) instead.",
        "lang.misc._block_dim",
    ),
    "TriMesh": (
        "Please import meshtaichi_patcher as Patcher and use Patcher.load_mesh(...) instead.",
        "lang.mesh._TriMesh",
    ),
    "TetMesh": (
        "Please import meshtaichi_patcher as Patcher and use Patcher.load_mesh(...) instead.",
        "lang.mesh._TetMesh",
    ),
}


def __getattr__(attr):
    import warnings  # pylint: disable=C0415,W0621

    if attr == "cfg":
        return None if lang.impl.get_runtime().prog is None else lang.impl.current_cfg()
    if attr in __deprecated_names__:
        warnings.warn(
            f"ti.{attr} is deprecated, and it will be removed in Taichi v1.6.0. Please use ti.{__deprecated_names__[attr]} instead.",
            DeprecationWarning,
        )
        exec(f"{attr} = {__deprecated_names__[attr]}")
        return locals()[attr]
    if attr in __customized_deprecations__:
        msg, fun = __customized_deprecations__[attr]
        warnings.warn(
            f"ti.{attr} is deprecated, and it will be removed in Taichi v1.6.0. {msg}",
            DeprecationWarning,
        )
        exec(f"{attr} = {fun}")
        return locals()[attr]
    raise AttributeError(f"module '{__name__}' has no attribute '{attr}'")


__version__ = (
    _ti_core.get_version_major(),
    _ti_core.get_version_minor(),
    _ti_core.get_version_patch(),
)

del sys
del _ti_core

warn_restricted_version()
del warn_restricted_version
