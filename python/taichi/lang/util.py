from .core import taichi_lang_core
from taichi.misc.util import warning, deprecated, obsolete, get_traceback
import numpy as np
import os

_has_pytorch = False

_env_torch = os.environ.get('TI_ENABLE_TORCH', '1')
if not _env_torch or int(_env_torch):
    try:
        import torch
        _has_pytorch = True
    except:
        pass


def has_pytorch():
    return _has_pytorch


def is_taichi_class(rhs):
    taichi_class = False
    try:
        if rhs.is_taichi_class:
            taichi_class = True
    except:
        pass
    return taichi_class


# Real types

float32 = taichi_lang_core.DataType_f32
f32 = float32
float64 = taichi_lang_core.DataType_f64
f64 = float64

real_types = [f32, f64, float]
real_type_ids = [id(t) for t in real_types]

# Integer types

int8 = taichi_lang_core.DataType_i8
i8 = int8
int16 = taichi_lang_core.DataType_i16
i16 = int16
int32 = taichi_lang_core.DataType_i32
i32 = int32
int64 = taichi_lang_core.DataType_i64
i64 = int64

uint8 = taichi_lang_core.DataType_u8
u8 = uint8
uint16 = taichi_lang_core.DataType_u16
u16 = uint16
uint32 = taichi_lang_core.DataType_u32
u32 = uint32
uint64 = taichi_lang_core.DataType_u64
u64 = uint64

integer_types = [i8, i16, i32, i64, u8, u16, u32, u64, int]
integer_type_ids = [id(t) for t in integer_types]

types = real_types + integer_types
type_ids = [id(t) for t in types]


def to_numpy_type(dt):
    if dt == f32:
        return np.float32
    elif dt == f64:
        return np.float64
    elif dt == i32:
        return np.int32
    elif dt == i64:
        return np.int64
    elif dt == i8:
        return np.int8
    elif dt == i16:
        return np.int16
    elif dt == u8:
        return np.uint8
    elif dt == u16:
        return np.uint16
    elif dt == u32:
        return np.uint32
    elif dt == u64:
        return np.uint64
    else:
        assert False


def to_pytorch_type(dt):
    import torch
    if dt == f32:
        return torch.float32
    elif dt == f64:
        return torch.float64
    elif dt == i32:
        return torch.int32
    elif dt == i64:
        return torch.int64
    elif dt == i8:
        return torch.int8
    elif dt == i16:
        return torch.int16
    elif dt == u8:
        return torch.uint8
    elif dt == u16:
        return torch.uint16
    elif dt == u32:
        return torch.uint32
    elif dt == u64:
        return torch.uint64
    else:
        assert False


def to_taichi_type(dt):
    if type(dt) == taichi_lang_core.DataType:
        return dt

    if dt == np.float32:
        return f32
    elif dt == np.float64:
        return f64
    elif dt == np.int32:
        return i32
    elif dt == np.int64:
        return i64
    elif dt == np.int8:
        return i8
    elif dt == np.int16:
        return i16
    elif dt == np.uint8:
        return u8
    elif dt == np.uint16:
        return u16
    elif dt == np.uint32:
        return u32
    elif dt == np.uint64:
        return u64

    if has_pytorch():
        if dt == torch.float32:
            return f32
        elif dt == torch.float64:
            return f64
        elif dt == torch.int32:
            return i32
        elif dt == torch.int64:
            return i64
        elif dt == torch.int8:
            return i8
        elif dt == torch.int16:
            return i16
        elif dt == torch.uint8:
            return u8
        elif dt == torch.uint16:
            return u16
        elif dt == torch.uint32:
            return u32
        elif dt == torch.uint64:
            return u64

    raise AssertionError("Unknown type {}".format(dt))


def cook_dtype(dtype):
    from .impl import get_runtime
    _taichi_skip_traceback = 1
    if isinstance(dtype, taichi_lang_core.DataType):
        return dtype
    elif isinstance(dtype, taichi_lang_core.Type):
        return taichi_lang_core.DataType(dtype)
    elif dtype is float:
        return get_runtime().default_fp
    elif dtype is int:
        return get_runtime().default_ip
    else:
        raise ValueError(f'Invalid data type {dtype}')


def in_taichi_scope():
    from . import impl
    return impl.inside_kernel()


def in_python_scope():
    return not in_taichi_scope()


def taichi_scope(func):
    import functools

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _taichi_skip_traceback = 1
        assert in_taichi_scope(), \
                f'{func.__name__} cannot be called in Python-scope'
        return func(*args, **kwargs)

    return wrapped


def python_scope(func):
    import functools

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        _taichi_skip_traceback = 1
        assert in_python_scope(), \
                f'{func.__name__} cannot be called in Taichi-scope'
        return func(*args, **kwargs)

    return wrapped
