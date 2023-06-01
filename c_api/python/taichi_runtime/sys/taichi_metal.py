import ctypes
from ._lib import _LIB
from .taichi_core import *


"""
Handle `TiNsBundle`
"""
TiNsBundle = ctypes.c_void_p


"""
Handle `TiMtlDevice`
"""
TiMtlDevice = ctypes.c_void_p


"""
Handle `TiMtlBuffer`
"""
TiMtlBuffer = ctypes.c_void_p


"""
Handle `TiMtlTexture`
"""
TiMtlTexture = ctypes.c_void_p


"""
Structure `TiMetalRuntimeInteropInfo`
"""


class TiMetalRuntimeInteropInfo(ctypes.Structure):
    pass


TiMetalRuntimeInteropInfo._fields_ = [
    ("bundle", TiNsBundle),
    ("device", TiMtlDevice),
]


"""
Structure `TiMetalMemoryInteropInfo`
"""


class TiMetalMemoryInteropInfo(ctypes.Structure):
    pass


TiMetalMemoryInteropInfo._fields_ = [
    ("buffer", TiMtlBuffer),
]


"""
Structure `TiMetalImageInteropInfo`
"""


class TiMetalImageInteropInfo(ctypes.Structure):
    pass


TiMetalImageInteropInfo._fields_ = [
    ("texture", TiMtlTexture),
]


_LIB.ti_import_metal_runtime.argtypes = [
    ctypes.c_void_p,  # const TiMetalRuntimeInteropInfo*,
]
_LIB.ti_import_metal_runtime.restype = TiRuntime


def ti_import_metal_runtime(
    interop_info: ctypes.c_void_p,  # const TiMetalRuntimeInteropInfo*
) -> TiRuntime:
    """
    Function `ti_import_metal_runtime`

    Return value: TiRuntime

    Parameters:
        interop_info (`TiMetalRuntimeInteropInfo`):
    """
    out = _LIB.ti_import_metal_runtime(interop_info)
    return TiRuntime(out)


_LIB.ti_export_metal_runtime.argtypes = [
    TiRuntime,
    ctypes.c_void_p,  # TiMetalRuntimeInteropInfo*,
]
_LIB.ti_export_metal_runtime.restype = None


def ti_export_metal_runtime(
    runtime: TiRuntime,
    interop_info: ctypes.c_void_p,  # TiMetalRuntimeInteropInfo*
) -> None:
    """
    Function `ti_export_metal_runtime`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiMetalRuntimeInteropInfo`):
    """
    out = _LIB.ti_export_metal_runtime(runtime, interop_info)


_LIB.ti_import_metal_memory.argtypes = [
    TiRuntime,
    ctypes.c_void_p,  # const TiMetalMemoryInteropInfo*,
]
_LIB.ti_import_metal_memory.restype = TiMemory


def ti_import_metal_memory(
    runtime: TiRuntime,
    interop_info: ctypes.c_void_p,  # const TiMetalMemoryInteropInfo*
) -> TiMemory:
    """
    Function `ti_import_metal_memory`

    Return value: TiMemory

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiMetalMemoryInteropInfo`):
    """
    out = _LIB.ti_import_metal_memory(runtime, interop_info)
    return TiMemory(out)


_LIB.ti_export_metal_memory.argtypes = [
    TiRuntime,
    TiMemory,
    ctypes.c_void_p,  # TiMetalMemoryInteropInfo*,
]
_LIB.ti_export_metal_memory.restype = None


def ti_export_metal_memory(
    runtime: TiRuntime,
    memory: TiMemory,
    interop_info: ctypes.c_void_p,  # TiMetalMemoryInteropInfo*
) -> None:
    """
    Function `ti_export_metal_memory`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        memory (`TiMemory`):
        interop_info (`TiMetalMemoryInteropInfo`):
    """
    out = _LIB.ti_export_metal_memory(runtime, memory, interop_info)


_LIB.ti_import_metal_image.argtypes = [
    TiRuntime,
    ctypes.c_void_p,  # const TiMetalImageInteropInfo*,
]
_LIB.ti_import_metal_image.restype = TiImage


def ti_import_metal_image(
    runtime: TiRuntime,
    interop_info: ctypes.c_void_p,  # const TiMetalImageInteropInfo*
) -> TiImage:
    """
    Function `ti_import_metal_image`

    Return value: TiImage

    Parameters:
        runtime (`TiRuntime`):
        interop_info (`TiMetalImageInteropInfo`):
    """
    out = _LIB.ti_import_metal_image(runtime, interop_info)
    return TiImage(out)


_LIB.ti_export_metal_image.argtypes = [
    TiRuntime,
    TiImage,
    ctypes.c_void_p,  # TiMetalImageInteropInfo*,
]
_LIB.ti_export_metal_image.restype = None


def ti_export_metal_image(
    runtime: TiRuntime,
    image: TiImage,
    interop_info: ctypes.c_void_p,  # TiMetalImageInteropInfo*
) -> None:
    """
    Function `ti_export_metal_image`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        image (`TiImage`):
        interop_info (`TiMetalImageInteropInfo`):
    """
    out = _LIB.ti_export_metal_image(runtime, image, interop_info)
