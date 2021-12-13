from math import acos, asin, cos, sin

from taichi._lib import core as _ti_core
from taichi.lang.impl import default_cfg
from taichi.lang.matrix import Vector


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
    elif default_cfg().arch == _ti_core.vulkan:
        info.field_source = _ti_core.FieldSource.TaichiVulkan
    else:
        raise Exception("unsupported taichi backend")
    info.shape = [n for n in field.shape]

    info.dtype = field.dtype
    info.snode = field.snode.ptr

    if hasattr(field, 'n'):
        info.field_type = _ti_core.FieldType.Matrix
        info.matrix_rows = field.n
        info.matrix_cols = field.m
    else:
        info.field_type = _ti_core.FieldType.Scalar
        info.matrix_rows = 1
        info.matrix_cols = 1
    return info


def euler_to_vec(yaw, pitch):
    v = Vector([0.0, 0.0, 0.0])
    v[0] = -sin(yaw) * cos(pitch)
    v[1] = sin(pitch)
    v[2] = -cos(yaw) * cos(pitch)
    return v


def vec_to_euler(v):
    v = v.normalized()
    pitch = asin(v[1])

    sin_yaw = -v[0] / cos(pitch)
    cos_yaw = -v[2] / cos(pitch)

    eps = 1e-6

    if abs(sin_yaw) < eps:
        yaw = 0
    else:
        yaw = acos(cos_yaw)
        if sin_yaw < 0:
            yaw = -yaw

    return yaw, pitch
