import ctypes
from ._lib import _LIB
from .taichi_core import *


"""
Structure `TiCpuMemoryInteropInfo`
"""


class TiCpuMemoryInteropInfo(ctypes.Structure):
    pass


TiCpuMemoryInteropInfo._fields_ = [
    ("ptr", ctypes.c_void_p),
    ("size", ctypes.c_uint64),
]


_LIB.ti_export_cpu_memory.argtypes = [
    TiRuntime,
    TiMemory,
    ctypes.c_void_p,  # TiCpuMemoryInteropInfo*
]
_LIB.ti_export_cpu_memory.restype = None


def ti_export_cpu_memory(
    runtime: TiRuntime,
    memory: TiMemory,
    interop_info: ctypes.c_void_p,  # TiCpuMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_cpu_memory`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiCpuMemoryInteropInfo`):
    """
    out = _LIB.ti_export_cpu_memory(runtime, memory, interop_info)
