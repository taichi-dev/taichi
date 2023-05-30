import ctypes
from ._lib import _LIB
from .taichi_core import *


"""
Handle `TixNativeBufferUnity`
"""
TixNativeBufferUnity = ctypes.c_void_p


"""
Callback `TixAsyncTaskUnity`
"""
TixAsyncTaskUnity = ctypes.c_void_p


_LIB.tix_import_native_runtime_unity.argtypes = [
]
_LIB.tix_import_native_runtime_unity.restype = TiRuntime
def tix_import_native_runtime_unity(
) -> TiRuntime:
    """
    Function `tix_import_native_runtime_unity`

    Return value: TiRuntime
    """
    return _LIB.tix_import_native_runtime_unity()


_LIB.tix_enqueue_task_async_unity.argtypes = [
    ctypes.c_void_p,
    TixAsyncTaskUnity,
]
_LIB.tix_enqueue_task_async_unity.restype = None
def tix_enqueue_task_async_unity(
  user_data: ctypes.c_void_p,
  async_task: TixAsyncTaskUnity
) -> None:
    """
    Function `tix_enqueue_task_async_unity`

    Return value: None

    Parameters:
        user_data (`ctypes.c_void_p`):
        async_task (`TixAsyncTaskUnity`):
    """
    return _LIB.tix_enqueue_task_async_unity(user_data, async_task)


_LIB.tix_launch_kernel_async_unity.argtypes = [
    TiRuntime,
    TiKernel,
    ctypes.c_uint32,
    ctypes.c_void_p, # const TiArgument*,
]
_LIB.tix_launch_kernel_async_unity.restype = None
def tix_launch_kernel_async_unity(
  runtime: TiRuntime,
  kernel: TiKernel,
  arg_count: ctypes.c_uint32,
  args: ctypes.c_void_p, # const TiArgument*
) -> None:
    """
    Function `tix_launch_kernel_async_unity`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        kernel (`TiKernel`):
        arg_count (`ctypes.c_uint32`):
        args (`TiArgument`):
    """
    return _LIB.tix_launch_kernel_async_unity(runtime, kernel, arg_count, args)


_LIB.tix_launch_compute_graph_async_unity.argtypes = [
    TiRuntime,
    TiComputeGraph,
    ctypes.c_uint32,
    ctypes.c_void_p, # const TiNamedArgument*,
]
_LIB.tix_launch_compute_graph_async_unity.restype = None
def tix_launch_compute_graph_async_unity(
  runtime: TiRuntime,
  compute_graph: TiComputeGraph,
  arg_count: ctypes.c_uint32,
  args: ctypes.c_void_p, # const TiNamedArgument*
) -> None:
    """
    Function `tix_launch_compute_graph_async_unity`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        compute_graph (`TiComputeGraph`):
        arg_count (`ctypes.c_uint32`):
        args (`TiNamedArgument`):
    """
    return _LIB.tix_launch_compute_graph_async_unity(runtime, compute_graph, arg_count, args)


_LIB.tix_copy_memory_to_native_buffer_async_unity.argtypes = [
    TiRuntime,
    TixNativeBufferUnity,
    ctypes.c_uint64,
    ctypes.c_void_p, # const TiMemorySlice*,
]
_LIB.tix_copy_memory_to_native_buffer_async_unity.restype = None
def tix_copy_memory_to_native_buffer_async_unity(
  runtime: TiRuntime,
  dst: TixNativeBufferUnity,
  dst_offset: ctypes.c_uint64,
  src: ctypes.c_void_p, # const TiMemorySlice*
) -> None:
    """
    Function `tix_copy_memory_to_native_buffer_async_unity`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        dst (`TixNativeBufferUnity`):
        dst_offset (`ctypes.c_uint64`):
        src (`TiMemorySlice`):
    """
    return _LIB.tix_copy_memory_to_native_buffer_async_unity(runtime, dst, dst_offset, src)


_LIB.tix_copy_memory_device_to_host_unity.argtypes = [
    TiRuntime,
    ctypes.c_void_p,
    ctypes.c_uint64,
    ctypes.c_void_p, # const TiMemorySlice*,
]
_LIB.tix_copy_memory_device_to_host_unity.restype = None
def tix_copy_memory_device_to_host_unity(
  runtime: TiRuntime,
  dst: ctypes.c_void_p,
  dst_offset: ctypes.c_uint64,
  src: ctypes.c_void_p, # const TiMemorySlice*
) -> None:
    """
    Function `tix_copy_memory_device_to_host_unity`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        dst (`ctypes.c_void_p`):
        dst_offset (`ctypes.c_uint64`):
        src (`TiMemorySlice`):
    """
    return _LIB.tix_copy_memory_device_to_host_unity(runtime, dst, dst_offset, src)


_LIB.tix_copy_memory_host_to_device_unity.argtypes = [
    TiRuntime,
    ctypes.c_void_p, # const TiMemorySlice*,
    ctypes.c_void_p,
    ctypes.c_uint64,
]
_LIB.tix_copy_memory_host_to_device_unity.restype = None
def tix_copy_memory_host_to_device_unity(
  runtime: TiRuntime,
  dst: ctypes.c_void_p, # const TiMemorySlice*,
  src: ctypes.c_void_p,
  src_offset: ctypes.c_uint64
) -> None:
    """
    Function `tix_copy_memory_host_to_device_unity`

    Return value: None

    Parameters:
        runtime (`TiRuntime`):
        dst (`TiMemorySlice`):
        src (`ctypes.c_void_p`):
        src_offset (`ctypes.c_uint64`):
    """
    return _LIB.tix_copy_memory_host_to_device_unity(runtime, dst, src, src_offset)


_LIB.tix_submit_async_unity.argtypes = [
    TiRuntime,
]
_LIB.tix_submit_async_unity.restype = ctypes.c_void_p
def tix_submit_async_unity(
  runtime: TiRuntime
) -> ctypes.c_void_p:
    """
    Function `tix_submit_async_unity`

    Return value: ctypes.c_void_p

    Parameters:
        runtime (`TiRuntime`):
    """
    return _LIB.tix_submit_async_unity(runtime)
