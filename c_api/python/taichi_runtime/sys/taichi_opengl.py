import ctypes
from ._lib import _LIB
from .taichi_core import *


"""
Structure `TiOpenglRuntimeInteropInfo`
"""
class TiOpenglRuntimeInteropInfo(ctypes.Structure): pass
TiOpenglRuntimeInteropInfo._fields_ = [
    ('get_proc_addr', ctypes.c_void_p),
]


_LIB.ti_import_opengl_runtime.argtypes = [
    ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*
]
_LIB.ti_import_opengl_runtime.restype = TiRuntime
def ti_import_opengl_runtime(
  interop_info: ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*
) -> TiRuntime:
    """
    Function `ti_import_opengl_runtime`

    Return value: TiRuntime

    Parameters:
        interop_info (`TiOpenglRuntimeInteropInfo`):
    """
    out = _LIB.ti_import_opengl_runtime(interop_info)
    return TiRuntime(out)


_LIB.ti_export_opengl_runtime.argtypes = [
    TiRuntime,
    ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*
]
_LIB.ti_export_opengl_runtime.restype = None
def ti_export_opengl_runtime(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*
) -> None:
    """
    Function `ti_export_opengl_runtime`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiOpenglRuntimeInteropInfo`):
    """
    out = _LIB.ti_export_opengl_runtime(runtime, interop_info)
