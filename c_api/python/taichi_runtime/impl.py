from enum import Enum
from functools import reduce
from typing import Any, ByteString, Dict, Iterable, List, Optional, Sequence, Tuple
from .sys.taichi_core import *
import numpy as np
import numpy.typing as npt
import warnings


def _p(x) -> ctypes.c_void_p:
    if isinstance(x, ByteString):
        return ctypes.cast(x, ctypes.c_void_p)
    else:
        return ctypes.c_void_p(ctypes.addressof(x))


def _null() -> ctypes.c_void_p:
    return ctypes.c_void_p(0)


class Error(Enum):
    SUCCESS = TI_ERROR_SUCCESS.value
    NOT_SUPPORTED = TI_ERROR_NOT_SUPPORTED.value
    CURROPTED_DATA = TI_ERROR_CORRUPTED_DATA.value
    NAME_NOT_FOUND = TI_ERROR_NAME_NOT_FOUND.value
    INVALID_ARGUMENT = TI_ERROR_INVALID_ARGUMENT.value
    ARGUMENT_NULL = TI_ERROR_ARGUMENT_NULL.value
    ARGUMENT_OUT_OF_RANGE = TI_ERROR_ARGUMENT_OUT_OF_RANGE.value
    ARGUMENT_NOT_FOUND = TI_ERROR_ARGUMENT_NOT_FOUND.value
    INVALID_INTEROP = TI_ERROR_INVALID_INTEROP.value
    INVALID_STATE = TI_ERROR_INVALID_STATE.value
    INCOMPATIBLE_MODULE = TI_ERROR_INCOMPATIBLE_MODULE.value
    OUT_OF_MEMORY = TI_ERROR_OUT_OF_MEMORY.value


class TaichiRuntimeError(Exception):
    def __init__(self, code: Error, message: str):
        super().__init__(code, message)

    def __str__(self):
        if self.message == "":
            return f"({self.code.name})"
        else:
            return f"({self.code.name}) {self.message}"

    @property
    def code(self) -> Error:
        return self.args[0]

    @property
    def message(self) -> str:
        return self.args[1]


def check_last_error():
    message_size = ctypes.c_uint32(0)
    code = ti_get_last_error(_p(message_size), _null())
    message = ctypes.create_string_buffer(message_size.value)
    ti_get_last_error(_p(message_size), _p(message))
    ti_set_last_error(TI_ERROR_SUCCESS, _null())

    if code.value != TI_ERROR_SUCCESS.value:
        raise TaichiRuntimeError(Error(code.value), message.value.decode("utf-8"))


class Arch(Enum):
    VULKAN = TI_ARCH_VULKAN.value
    METAL = TI_ARCH_METAL.value
    CUDA = TI_ARCH_CUDA.value
    X64 = TI_ARCH_X64.value
    ARM64 = TI_ARCH_ARM64.value
    OPENGL = TI_ARCH_OPENGL.value
    GLES = TI_ARCH_GLES.value


class Runtime:
    def __init__(self, handle: TiRuntime, *, should_destroy=True) -> None:
        self._handle = handle
        self._should_destroy = should_destroy

    def __del__(self):
        self.destroy(quiet=True)

    @staticmethod
    def create(arch: Arch | List[Arch], *, device_index: int = 0) -> 'Runtime':
        if isinstance(arch, Arch):
            arch = [arch]
        handle= TiRuntime(0)
        for a in arch:
            try:
                handle = ti_create_runtime(TiArch(a.value), ctypes.c_uint32(device_index))
                check_last_error()
                break
            except:
                continue
        if handle.value == 0:
            raise TaichiRuntimeError(Error.NOT_SUPPORTED, "No supported arch is found.")
        return Runtime(handle)

    def destroy(self, *, quiet: bool = False):
        if self._should_destroy:
            if self._handle.value == 0 and not quiet:
                if not quiet:
                    warnings.warn("Runtime.destroy() is called on a null handle.")
            else:
                ti_destroy_runtime(self._handle)
                check_last_error()
                self._should_destroy = False
                self._handle = TiRuntime(0)

    def wait(self):
        ti_wait(self._handle)
        check_last_error()

    def copy_memory_device_to_device(self, *, dst: 'Memory', src: 'Memory'):
        dst2 = TiMemorySlice(
            memory=dst._handle,
            offset=0,
            size=dst._size,
        )
        src2 = TiMemorySlice(
            memory=src._handle,
            offset=0,
            size=src._size,
        )
        ti_copy_memory_device_to_device(self._handle, _p(dst2), _p(src2))
        check_last_error()


class MemoryUsage(Enum):
    STORAGE = TI_MEMORY_USAGE_STORAGE_BIT.value
    UNIFORM = TI_MEMORY_USAGE_UNIFORM_BIT.value
    VERTEX = TI_MEMORY_USAGE_VERTEX_BIT.value
    INDEX = TI_MEMORY_USAGE_INDEX_BIT.value


class Memory:
    def __init__(self, runtime: Runtime, handle: TiMemory, *, size: int, host_access: bool, should_destroy: bool = True):
        self._runtime = runtime
        self._handle = handle
        self._size = size
        self._host_access = host_access
        self._should_destroy = should_destroy

    def __del__(self):
        self.free(quiet=True)

    @property
    def size(self) -> int:
        return self._size

    @property
    def host_access(self) -> bool:
        return self._host_access

    @staticmethod
    def allocate(runtime: Runtime, *, size: int, host_access: bool = False, usage: MemoryUsage):
        allocate_info = TiMemoryAllocateInfo(
            size=ctypes.c_uint64(size),
            host_write=TI_TRUE if host_access else TI_FALSE,
            host_read=TI_TRUE if host_access else TI_FALSE,
            export_sharing=TI_FALSE,
            usage=usage.value,
        )
        handle = ti_allocate_memory(runtime._handle, _p(allocate_info))
        check_last_error()
        return Memory(runtime, handle, size=size, host_access=host_access)

    def free(self, *, quiet: bool = False):
        if self._should_destroy:
            if self._handle.value == 0:
                if not quiet:
                    warnings.warn("Memory.free() is called on a null handle.")
            else:
                ti_free_memory(self._runtime._handle, self._handle)
                check_last_error()
                self._should_destroy = False
                self._handle = TiMemory(0)

    def read(self, dst: ByteString, *, force: bool = False):
        assert isinstance(dst, ByteString)
        assert len(dst) == self._size, f"len(dst) ({len(dst)}) != self._size ({self._size})"

        if self._host_access:
            mapped = ti_map_memory(self._runtime._handle, self._handle)
            check_last_error()
            assert mapped.value is not None

            src = (ctypes.c_byte * self._size).from_address(mapped.value)
            ctypes.memmove(_p(dst), _p(src), self._size)

            ti_unmap_memory(self._runtime._handle, self._handle)
            check_last_error()
        elif force:
            staging_buffer = Memory.allocate(self._runtime, size=self._size, host_access=True, usage=MemoryUsage.STORAGE)
            self._runtime.copy_memory_device_to_device(dst=staging_buffer, src=self)
            self._runtime.wait()
            staging_buffer.read(dst)
            del staging_buffer
        else:
            raise TaichiRuntimeError(Error.NOT_SUPPORTED, "Memory.read() is not supported when `host_access` is False. Use `force=True` to force copying to host.")

    def write(self, src: ByteString, *, force: bool = False):
        assert isinstance(src, ByteString)
        assert len(src) == self._size, f"len(src) ({len(src)}) != self._size ({self._size})"

        if self._host_access:
            mapped = ti_map_memory(self._runtime._handle, self._handle)
            check_last_error()
            assert mapped.value is not None

            dst = (ctypes.c_byte * self._size).from_address(mapped.value)
            ctypes.memmove(_p(dst), _p(src), self._size)

            ti_unmap_memory(self._runtime._handle, self._handle)
            check_last_error()
        elif force:
            staging_buffer = Memory.allocate(self._runtime, size=self._size, host_access=True, usage=MemoryUsage.STORAGE)
            staging_buffer.write(src)
            self._runtime.copy_memory_device_to_device(dst=self, src=staging_buffer)
            self._runtime.wait()
            del staging_buffer
        else:
            raise TaichiRuntimeError(Error.NOT_SUPPORTED, "Memory.write() is not supported when `host_access` is False. Use `force=True` to force copying to host.")

    @staticmethod
    def from_bytes(runtime: Runtime, src: ByteString, *, host_access: bool = False):
        assert isinstance(src, ByteString)

        memory = Memory.allocate(runtime, size=len(src), host_access=host_access, usage=MemoryUsage.STORAGE)
        memory.write(src, force=True)
        return memory

    def to_bytes(self) -> bytes:
        dst = bytes(self._size)
        self.read(dst, force=True)
        return dst


class DataType(Enum):
    F16 = TI_DATA_TYPE_F16.value
    F32 = TI_DATA_TYPE_F32.value
    F64 = TI_DATA_TYPE_F64.value
    I8 = TI_DATA_TYPE_I8.value
    I16 = TI_DATA_TYPE_I16.value
    I32 = TI_DATA_TYPE_I32.value
    I64 = TI_DATA_TYPE_I64.value
    U8 = TI_DATA_TYPE_U8.value
    U16 = TI_DATA_TYPE_U16.value
    U32 = TI_DATA_TYPE_U32.value
    U64 = TI_DATA_TYPE_U64.value


_DTYPE_SIZE_TABLE = {
    TI_DATA_TYPE_F16.value: 2,
    TI_DATA_TYPE_F32.value: 4,
    TI_DATA_TYPE_F64.value: 8,
    TI_DATA_TYPE_I8.value: 1,
    TI_DATA_TYPE_I16.value: 2,
    TI_DATA_TYPE_I32.value: 4,
    TI_DATA_TYPE_I64.value: 8,
    TI_DATA_TYPE_U8.value: 1,
    TI_DATA_TYPE_U16.value: 2,
    TI_DATA_TYPE_U32.value: 4,
    TI_DATA_TYPE_U64.value: 8,
}

_NP_DTYPE_TABLE: Dict[str, DataType] = {
    'float16': DataType.F16,
    'float32': DataType.F32,
    'float64': DataType.F64,
    'int8': DataType.I8,
    'int16': DataType.I16,
    'int32': DataType.I32,
    'int64': DataType.I64,
    'uint8': DataType.U8,
    'uint16': DataType.U16,
    'uint32': DataType.U32,
    'uint64': DataType.U64,
}
_DTYPE_NP_TABLE: Dict[DataType, type] = {
    DataType.F16: np.float16,
    DataType.F32: np.float32,
    DataType.F64: np.float64,
    DataType.I8: np.int8,
    DataType.I16: np.int16,
    DataType.I32: np.int32,
    DataType.I64: np.int64,
    DataType.U8: np.uint8,
    DataType.U16: np.uint16,
    DataType.U32: np.uint32,
    DataType.U64: np.uint64,
}


class NdArray:
    def __init__(self, runtime: Runtime, memory: Memory, *, shape: Tuple[int], elem_shape: Tuple[int], elem_type: DataType):
        self._runtime = runtime
        self._memory = memory
        self._shape = shape
        self._elem_shape = elem_shape
        self._elem_type = elem_type

    def __del__(self):
        self.free()

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def shape(self) -> Tuple[int]:
        return self._shape

    @property
    def elem_shape(self) -> Tuple[int]:
        return self._elem_shape

    @property
    def elem_type(self) -> DataType:
        return self._elem_type

    @staticmethod
    def allocate(runtime: Runtime, elem_type: DataType, *, shape: Iterable[int], elem_shape: Iterable[int], host_access: bool = False):
        size = reduce(lambda x, y: x * y, shape, 1) * reduce(lambda x, y: x * y, elem_shape, 1) * _DTYPE_SIZE_TABLE[elem_type.value]
        memory = Memory.allocate(runtime, size=size, host_access=host_access, usage=MemoryUsage.STORAGE)
        return NdArray(runtime, memory, shape=tuple(shape), elem_shape=tuple(elem_shape), elem_type=elem_type)

    def free(self):
        self._memory.free()

    @staticmethod
    def from_numpy(runtime: Runtime, arr: npt.NDArray[Any], *, shape: Optional[Iterable[int]] = None, elem_shape: Optional[Iterable[int]] = None, elem_type: Optional[DataType] = None, host_access=False):
        assert isinstance(arr, np.ndarray)

        if elem_type is None:
            elem_type2 = _NP_DTYPE_TABLE[arr.dtype.name]
        else:
            elem_type2 = elem_type

        if elem_shape is None:
            elem_shape2: Tuple[int] = tuple()
        else:
            elem_shape2 = tuple(elem_shape)
            assert len(elem_shape2) <= len(arr.shape)
            for ielem_shape, ishape in enumerate(range(len(arr.shape) - len(elem_shape2), len(arr.shape))):
                assert arr.shape[ishape] == elem_shape2[ielem_shape], f"arr.shape[{ishape}] ({arr.shape[ishape]}) != elem_shape2[{ielem_shape}] ({elem_shape2[ielem_shape]})"

        if shape is None:
            shape2 = arr.shape[:len(arr.shape) - len(elem_shape2)]
        else:
            shape2 = tuple(shape)
            assert len(shape2) <= len(arr.shape)
            for i in range(len(shape2)):
                assert arr.shape[i] == shape2[i]

        memory = Memory.from_bytes(runtime, arr.tobytes(), host_access=host_access)
        return NdArray(runtime, memory, shape=shape2, elem_shape=elem_shape2, elem_type=elem_type2)

    def to_numpy(self) -> npt.NDArray[Any]:
        out = np.frombuffer(self.memory.to_bytes(), dtype=_DTYPE_NP_TABLE[self.elem_type]).reshape(self.shape + self.elem_shape)
        return out

    def into_numpy(self) -> npt.NDArray[Any]:
        out = self.to_numpy()
        self.free()
        return out


class ArgumentType(Enum):
    I32 = TI_ARGUMENT_TYPE_I32.value
    F32 = TI_ARGUMENT_TYPE_F32.value
    NDARRAY = TI_ARGUMENT_TYPE_NDARRAY.value


class Argument:
    def __init__(self, value: Any, *, ty: Optional[ArgumentType] = None) -> None:
        if isinstance(value, int):
            ty = ArgumentType.I32
        elif isinstance(value, float):
            ty = ArgumentType.F32
        elif isinstance(value, NdArray):
            ty = ArgumentType.NDARRAY
        else:
            raise TaichiRuntimeError(Error.NOT_SUPPORTED, f"{type(value)} is not a valid argument type.")

        if ty == ArgumentType.I32:
            assert isinstance(value, int)
            value = TiArgumentValue(
                i32 = ctypes.c_int32(value),
            )
        elif ty == ArgumentType.F32:
            assert isinstance(value, float)
            value = TiArgumentValue(
                f32 = ctypes.c_float(value),
            )
        elif ty == ArgumentType.NDARRAY:
            assert isinstance(value, NdArray)
            shape = TiNdShape(
                dim_count = ctypes.c_uint32(len(value._shape)),
                dims = (ctypes.c_uint32 * 16)(*[ctypes.c_uint32(x) for x in value._shape] + [0] * (16 - len(value._shape))),
            )
            elem_shape = TiNdShape(
                dim_count = ctypes.c_uint32(len(value._elem_shape)),
                dims = (ctypes.c_uint32 * 16)(*[ctypes.c_uint32(x) for x in value._elem_shape] + [0] * (16 - len(value._elem_shape))),
            )
            x = TiNdArray(
                memory = value._memory._handle,
                shape = shape,
                elem_shape = elem_shape,
                elem_type = value._elem_type.value,
            )
            value = TiArgumentValue(
                ndarray = x,
            )
        else:
            raise TaichiRuntimeError(Error.NOT_SUPPORTED, f"ArgumentType.{ty.name} is not supported.")

        self._ty = ty
        self._value = value


class Kernel:
    def __init__(self, aot_module: 'AotModule', name: str, handle: TiAotModule):
        self._aot_module = aot_module
        self._name = name
        self._handle = handle

    def __call__(self, *args: Any):
        self.launch(*args)

    def launch(self, *args: Any):
        args2 = [arg if isinstance(arg, Argument) else Argument(arg) for arg in args]
        args3 = [TiArgument(type=arg._ty.value, value=arg._value) for arg in args2]
        args4 = (TiArgument * len(args3))(*args3)
        ti_launch_kernel(self._aot_module._runtime._handle, self._handle, ctypes.c_uint32(len(args3)), _p(args4))
        check_last_error()


class AotModule:
    def __init__(self, runtime: Runtime, handle: TiAotModule, *, should_destroy: bool = True):
        self._runtime = runtime
        self._handle = handle
        self._should_destroy = should_destroy

    def __del__(self):
        self.destroy(quiet=True)

    @staticmethod
    def load(runtime: Runtime, path: str):
        handle = ti_load_aot_module(runtime._handle, _p(path.encode("ascii")))
        check_last_error()
        return AotModule(runtime, handle)
    
    @staticmethod
    def create(runtime: Runtime, tcm: bytes):
        handle = ti_create_aot_module(runtime._handle, _p(tcm), ctypes.c_uint64(len(tcm)))
        check_last_error()
        return AotModule(runtime, handle)

    def destroy(self, *, quiet: bool = False):
        if self._should_destroy:
            if self._handle.value == 0:
                if not quiet:
                    warnings.warn("AotModule.destroy() is called on a null handle.")
            else:
                ti_destroy_aot_module(self._handle)
                check_last_error()
                self._should_destroy = False
                self._handle = TiAotModule(0)

    def get_kernel(self, name: str) -> Kernel:
        handle = ti_get_aot_module_kernel(self._handle, _p(name.encode("utf-8")))
        check_last_error()
        return Kernel(self, name, handle)
