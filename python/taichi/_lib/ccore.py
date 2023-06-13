import os
import sys
import ctypes

from taichi._lib.utils import package_root


def _load_dll(path):
    try:
        if (
            sys.version_info[0] > 3
            or sys.version_info[0] == 3
            and sys.version_info[1] >= 8
        ):
            dll = ctypes.CDLL(path, winmode=0)
        else:
            dll = ctypes.CDLL(path)
    except OSError as ex:
        raise ex from None
    return dll


def load_core_exports_dll():
    bin_path = os.path.join(package_root, "_lib", "core_exports")
    if os.name == "nt":
        if (
            sys.version_info[0] > 3
            or sys.version_info[0] == 3
            and sys.version_info[1] >= 8
        ):
            os.add_dll_directory(bin_path)
        else:
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        dll_path = os.path.join(bin_path, "taichi_core_exports.dll")
    elif sys.platform == "darwin":
        dll_path = os.path.join(bin_path, "libtaichi_core_exports.dylib")
    else:
        dll_path = os.path.join(bin_path, "libtaichi_core_exports.so")

    return _load_dll(dll_path)


class TaichiCCore:
    def __init__(self) -> None:
        self._dll = load_core_exports_dll()
        if self._dll is None:
            raise RuntimeError("Cannot load taichi_core_exports.dll")

        # int ticore_hello_world(const char *extra_msg);
        self._dll.ticore_hello_world.argtypes = [ctypes.c_char_p]
        self._dll.ticore_hello_world.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_create(TieKernelHandle kernel_handle, TieLaunchContextBuilderHandle *ret_handle);
        self._dll.tie_LaunchContextBuilder_create.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self._dll.tie_LaunchContextBuilder_create.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_destroy(TieLaunchContextBuilderHandle handle);
        self._dll.tie_LaunchContextBuilder_destroy.argtypes = [ctypes.c_void_p]
        self._dll.tie_LaunchContextBuilder_destroy.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_arg_int(TieLaunchContextBuilderHandle handle, int arg_id, int64_t i64);
        self._dll.tie_LaunchContextBuilder_set_arg_int.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int64,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_int.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_arg_uint(TieLaunchContextBuilderHandle handle, int arg_id, uint64_t u64);
        self._dll.tie_LaunchContextBuilder_set_arg_uint.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_uint.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_arg_float(TieLaunchContextBuilderHandle handle, int arg_id, double d);
        self._dll.tie_LaunchContextBuilder_set_arg_float.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_float.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_struct_arg_int(TieLaunchContextBuilderHandle handle, int *arg_indices, size_t arg_indices_dim, int64_t i64);
        self._dll.tie_LaunchContextBuilder_set_struct_arg_int.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.c_int64,
        ]
        self._dll.tie_LaunchContextBuilder_set_struct_arg_int.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_struct_arg_uint(TieLaunchContextBuilderHandle handle, int *arg_indices, size_t arg_indices_dim, uint64_t u64);
        self._dll.tie_LaunchContextBuilder_set_struct_arg_uint.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        self._dll.tie_LaunchContextBuilder_set_struct_arg_uint.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_struct_arg_float(TieLaunchContextBuilderHandle handle, int *arg_indices, size_t arg_indices_dim, double d);
        self._dll.tie_LaunchContextBuilder_set_struct_arg_float.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.c_double,
        ]
        self._dll.tie_LaunchContextBuilder_set_struct_arg_float.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_arg_external_array_with_shape(TieLaunchContextBuilderHandle handle, int arg_id, uintptr_t ptr, uint64_t size, int64_t *shape, size_t shape_dim, uintptr_t grad_ptr);
        self._dll.tie_LaunchContextBuilder_set_arg_external_array_with_shape.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_size_t,
            ctypes.c_uint64,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_external_array_with_shape.restype = (
            ctypes.c_int
        )

        # int tie_LaunchContextBuilder_set_arg_ndarray(TieLaunchContextBuilderHandle handle, int arg_id, TieNdarrayHandle arr);
        self._dll.tie_LaunchContextBuilder_set_arg_ndarray.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_ndarray.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_arg_ndarray_with_grad(TieLaunchContextBuilderHandle handle, int arg_id, TieNdarrayHandle arr, TieNdarrayHandle arr_grad);
        self._dll.tie_LaunchContextBuilder_set_arg_ndarray_with_grad.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_ndarray_with_grad.restype = (
            ctypes.c_int
        )

        # int tie_LaunchContextBuilder_set_arg_texture(TieLaunchContextBuilderHandle handle, int arg_id, TieTextureHandle tex);
        self._dll.tie_LaunchContextBuilder_set_arg_texture.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_texture.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_set_arg_rw_texture(TieLaunchContextBuilderHandle handle, int arg_id, TieTextureHandle tex);
        self._dll.tie_LaunchContextBuilder_set_arg_rw_texture.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self._dll.tie_LaunchContextBuilder_set_arg_rw_texture.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_get_struct_ret_int(TieLaunchContextBuilderHandle handle, int *index, size_t index_dim, int64_t *ret_i64);
        self._dll.tie_LaunchContextBuilder_get_struct_ret_int.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_int64),
        ]
        self._dll.tie_LaunchContextBuilder_get_struct_ret_int.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_get_struct_ret_uint(TieLaunchContextBuilderHandle handle, int *index, size_t index_dim, uint64_t *ret_u64);
        self._dll.tie_LaunchContextBuilder_get_struct_ret_uint.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self._dll.tie_LaunchContextBuilder_get_struct_ret_uint.restype = ctypes.c_int

        # int tie_LaunchContextBuilder_get_struct_ret_float(TieLaunchContextBuilderHandle handle, int *index, size_t index_dim, double *ret_d);
        self._dll.tie_LaunchContextBuilder_get_struct_ret_float.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._dll.tie_LaunchContextBuilder_get_struct_ret_float.restype = ctypes.c_int

    def hello_world(self, extra_msg):
        return self._dll.ticore_hello_world(extra_msg.encode("utf-8"))

    def tie_LaunchContextBuilder_create(self, kernel_handle: int) -> int:
        handle = ctypes.c_void_p()
        ret = self._dll.tie_LaunchContextBuilder_create(
            kernel_handle, ctypes.byref(handle)
        )
        if ret != 0:  # Temp
            raise RuntimeError("Failed to create LaunchContextBuilder")
        return handle.value

    def tie_LaunchContextBuilder_destroy(self, handle: int):
        ret = self._dll.tie_LaunchContextBuilder_destroy(handle)
        if ret != 0:
            raise RuntimeError("Failed to destroy LaunchContextBuilder")

    def tie_LaunchContextBuilder_set_arg_int(self, handle: int, arg_id: int, i64: int):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_int(handle, arg_id, i64)
        if ret != 0:
            raise RuntimeError("Failed to set arg int")

    def tie_LaunchContextBuilder_set_arg_uint(self, handle: int, arg_id: int, u64: int):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_uint(handle, arg_id, u64)
        if ret != 0:
            raise RuntimeError("Failed to set arg uint")

    def tie_LaunchContextBuilder_set_arg_float(
        self, handle: int, arg_id: int, d: float
    ):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_float(handle, arg_id, d)
        if ret != 0:
            raise RuntimeError("Failed to set arg float")

    def tie_LaunchContextBuilder_set_struct_arg_int(
        self, handle: int, arg_indices: list, i64: int
    ):
        arg_indices = (ctypes.c_int * len(arg_indices))(*arg_indices)
        ret = self._dll.tie_LaunchContextBuilder_set_struct_arg_int(
            handle, arg_indices, len(arg_indices), i64
        )
        if ret != 0:
            raise RuntimeError("Failed to set struct arg int")

    def tie_LaunchContextBuilder_set_struct_arg_uint(
        self, handle: int, arg_indices: list, u64: int
    ):
        arg_indices = (ctypes.c_int * len(arg_indices))(*arg_indices)
        ret = self._dll.tie_LaunchContextBuilder_set_struct_arg_uint(
            handle, arg_indices, len(arg_indices), u64
        )
        if ret != 0:
            raise RuntimeError("Failed to set struct arg uint")

    def tie_LaunchContextBuilder_set_struct_arg_float(
        self, handle: int, arg_indices: list, d: float
    ):
        arg_indices = (ctypes.c_int * len(arg_indices))(*arg_indices)
        ret = self._dll.tie_LaunchContextBuilder_set_struct_arg_float(
            handle, arg_indices, len(arg_indices), d
        )
        if ret != 0:
            raise RuntimeError("Failed to set struct arg float")

    def tie_LaunchContextBuilder_set_arg_external_array_with_shape(
        self, handle: int, arg_id: int, ptr: int, size: int, shape: list, grad_ptr: int
    ):
        shape = (ctypes.c_int64 * len(shape))(*shape)
        ret = self._dll.tie_LaunchContextBuilder_set_arg_external_array_with_shape(
            handle, arg_id, ptr, size, shape, len(shape), grad_ptr
        )
        if ret != 0:
            raise RuntimeError("Failed to set arg external array with shape")

    def tie_LaunchContextBuilder_set_arg_ndarray(
        self, handle: int, arg_id: int, arr: int
    ):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_ndarray(handle, arg_id, arr)
        if ret != 0:
            raise RuntimeError("Failed to set arg ndarray")

    def tie_LaunchContextBuilder_set_arg_ndarray_with_grad(
        self, handle: int, arg_id: int, arr: int, arr_grad: int
    ):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_ndarray_with_grad(
            handle, arg_id, arr, arr_grad
        )
        if ret != 0:
            raise RuntimeError("Failed to set arg ndarray with grad")

    def tie_LaunchContextBuilder_set_arg_texture(
        self, handle: int, arg_id: int, tex: int
    ):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_texture(handle, arg_id, tex)
        if ret != 0:
            raise RuntimeError("Failed to set arg texture")

    def tie_LaunchContextBuilder_set_arg_rw_texture(
        self, handle: int, arg_id: int, tex: int
    ):
        ret = self._dll.tie_LaunchContextBuilder_set_arg_rw_texture(handle, arg_id, tex)
        if ret != 0:
            raise RuntimeError("Failed to set arg rw texture")

    def tie_LaunchContextBuilder_get_struct_ret_int(self, handle: int, index: list):
        index = (ctypes.c_int * len(index))(*index)
        ret_i64 = ctypes.c_int64()
        ret = self._dll.tie_LaunchContextBuilder_get_struct_ret_int(
            handle, index, len(index), ctypes.byref(ret_i64)
        )
        if ret != 0:
            raise RuntimeError("Failed to get struct ret int")
        return ret_i64.value

    def tie_LaunchContextBuilder_get_struct_ret_uint(self, handle: int, index: list):
        index = (ctypes.c_int * len(index))(*index)
        ret_u64 = ctypes.c_uint64()
        ret = self._dll.tie_LaunchContextBuilder_get_struct_ret_uint(
            handle, index, len(index), ctypes.byref(ret_u64)
        )
        if ret != 0:
            raise RuntimeError("Failed to get struct ret uint")
        return ret_u64.value

    def tie_LaunchContextBuilder_get_struct_ret_float(self, handle: int, index: list):
        index = (ctypes.c_int * len(index))(*index)
        ret_d = ctypes.c_double()
        ret = self._dll.tie_LaunchContextBuilder_get_struct_ret_float(
            handle, index, len(index), ctypes.byref(ret_d)
        )
        if ret != 0:
            raise RuntimeError("Failed to get struct ret float")
        return ret_d.value


taichi_ccore = TaichiCCore()
