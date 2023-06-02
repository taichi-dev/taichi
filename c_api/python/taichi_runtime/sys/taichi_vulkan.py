"""
# Vulkan Backend Features

Taichi's Vulkan API gives you further control over the Vulkan version and extension requirements and allows you to interop with external Vulkan applications with shared resources.

"""
import ctypes
from ._lib import _LIB
from .taichi_core import *


_LIB.ti_create_vulkan_runtime_ext.argtypes = [
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_void_p,
]
_LIB.ti_create_vulkan_runtime_ext.restype = TiRuntime
def ti_create_vulkan_runtime_ext(
  api_version: ctypes.c_uint32,
  instance_extension_count: ctypes.c_uint32,
  instance_extensions: ctypes.c_void_p,
  device_extension_count: ctypes.c_uint32,
  device_extensions: ctypes.c_void_p,
) -> TiRuntime:
    """
    Function `ti_create_vulkan_runtime_ext` (1.4.0)
    
    Creates a Vulkan Taichi runtime with user-controlled capability settings.

    Return value: TiRuntime

    Parameters:
        api_version (`ctypes.c_uint32`):
        instance_extension_count (`ctypes.c_uint32`):
        instance_extensions (`ctypes.c_void_p`):
        device_extension_count (`ctypes.c_uint32`):
        device_extensions (`ctypes.c_void_p`):
    """
    out = _LIB.ti_create_vulkan_runtime_ext(api_version, instance_extension_count, instance_extensions, device_extension_count, device_extensions)
    return TiRuntime(out)
