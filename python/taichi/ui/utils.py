import pathlib

from taichi.core import ti_core as _ti_core
from taichi.core.primitive_types import *
from taichi.lang.impl import default_cfg
from taichi.lang.kernel_arguments import ext_arr, template
from taichi.lang.kernel_impl import kernel
from taichi.lang.ops import get_addr


@kernel
def get_field_addr_0D(x: template()) -> u64:
    return get_addr(x, [None])


@kernel
def get_field_addr_ND(x: template()) -> u64:
    return get_addr(x, [0 for _ in x.shape])


field_addr_cache = {}


def get_field_addr(x):
    if x not in field_addr_cache:
        if len(x.shape) == 0:
            addr = get_field_addr_0D(x)
        else:
            addr = get_field_addr_ND(x)
        field_addr_cache[x] = addr
    return field_addr_cache[x]


def get_field_info(field):
    info = _ti_core.FieldInfo()
    if field is None:
        info.valid = False
        return info
    info.valid = True
    if default_cfg().arch == _ti_core.cuda:
        info.field_source = _ti_core.FieldSource.TaichiCuda
    elif default_cfg().arch == _ti_core.x64:
        info.field_source = _ti_core.FieldSource.TaichiX64
    else:
        raise Exception("unsupported taichi backend")
    info.shape = [n for n in field.shape]

    info.dtype = field.dtype
    info.data = get_field_addr(field)

    if hasattr(field, 'n'):
        info.field_type = _ti_core.FieldType.Matrix
        info.matrix_rows = field.n
        info.matrix_cols = field.m
    else:
        info.field_type = _ti_core.FieldType.Scalar
    return info
