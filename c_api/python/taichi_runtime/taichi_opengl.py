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


"""
Structure `TiOpenglMemoryInteropInfo`
"""
class TiOpenglMemoryInteropInfo(ctypes.Structure): pass
TiOpenglMemoryInteropInfo._fields_ = [
    ('buffer', GLuint),
    ('size', GLsizeiptr),
]


"""
Structure `TiOpenglImageInteropInfo`
"""
class TiOpenglImageInteropInfo(ctypes.Structure): pass
TiOpenglImageInteropInfo._fields_ = [
    ('texture', GLuint),
    ('target', GLenum),
    ('levels', GLsizei),
    ('format', GLenum),
    ('width', GLsizei),
    ('height', GLsizei),
    ('depth', GLsizei),
]


_LIB.ti_import_opengl_runtime.argtypes = [
    ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*,
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
    return _LIB.ti_import_opengl_runtime(interop_info)


_LIB.ti_export_opengl_runtime.argtypes = [
    TiRuntime,
    ctypes.c_void_p, # TiOpenglRuntimeInteropInfo*,
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
    return _LIB.ti_export_opengl_runtime(runtime, interop_info)


_LIB.ti_import_opengl_memory.argtypes = [
    TiRuntime,
    ctypes.c_void_p, # const TiOpenglMemoryInteropInfo*,
]
_LIB.ti_import_opengl_memory.restype = TiMemory
def ti_import_opengl_memory(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # const TiOpenglMemoryInteropInfo*
) -> TiMemory:
    """
    Function `ti_import_opengl_memory`

    Return value: TiMemory

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiOpenglMemoryInteropInfo`):
    """
    return _LIB.ti_import_opengl_memory(runtime, interop_info)


_LIB.ti_export_opengl_memory.argtypes = [
    TiRuntime,
    TiMemory,
    ctypes.c_void_p, # TiOpenglMemoryInteropInfo*,
]
_LIB.ti_export_opengl_memory.restype = None
def ti_export_opengl_memory(
  runtime: TiRuntime,
  memory: TiMemory,
  interop_info: ctypes.c_void_p, # TiOpenglMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_opengl_memory`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiOpenglMemoryInteropInfo`):
    """
    return _LIB.ti_export_opengl_memory(runtime, memory, interop_info)


_LIB.ti_import_opengl_image.argtypes = [
    TiRuntime,
    ctypes.c_void_p, # const TiOpenglImageInteropInfo*,
]
_LIB.ti_import_opengl_image.restype = TiImage
def ti_import_opengl_image(
  runtime: TiRuntime,
  interop_info: ctypes.c_void_p, # const TiOpenglImageInteropInfo*
) -> TiImage:
    """
    Function `ti_import_opengl_image`

    Return value: TiImage

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiOpenglImageInteropInfo`):
    """
    return _LIB.ti_import_opengl_image(runtime, interop_info)


_LIB.ti_export_opengl_image.argtypes = [
    TiRuntime,
    TiImage,
    ctypes.c_void_p, # TiOpenglImageInteropInfo*,
]
_LIB.ti_export_opengl_image.restype = None
def ti_export_opengl_image(
  runtime: TiRuntime,
  image: TiImage,
  interop_info: ctypes.c_void_p, # TiOpenglImageInteropInfo*
) -> None:
    """
    Function `ti_export_opengl_image`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        interop_info (`TiOpenglImageInteropInfo`):
    """
    return _LIB.ti_export_opengl_image(runtime, image, interop_info)
