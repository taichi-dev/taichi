import ctypes
from ._lib import _LIB
from .taichi_core import *


"""
Structure `TiCudaMemoryInteropInfo`
"""
class TiCudaMemoryInteropInfo(ctypes.Structure): pass
TiCudaMemoryInteropInfo._fields_ = [
    ('ptr', ctypes.c_void_p),
    ('size', ctypes.c_uint64),
]


_LIB.ti_export_cuda_memory.argtypes = [
    TiRuntime,
    TiMemory,
    ctypes.c_void_p, # TiCudaMemoryInteropInfo*,
]
_LIB.ti_export_cuda_memory.restype = None
def ti_export_cuda_memory(
  runtime: TiRuntime,
  memory: TiMemory,
  interop_info: ctypes.c_void_p, # TiCudaMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_cuda_memory`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiCudaMemoryInteropInfo`):
    """
    out = _LIB.ti_export_cuda_memory(runtime, memory, interop_info)
