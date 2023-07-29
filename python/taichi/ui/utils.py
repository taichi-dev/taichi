import platform
from math import asin, cos, sin

import numpy as np
from taichi._lib import core as _ti_core
from taichi._lib.utils import try_get_wheel_tag
from taichi.lang._ndarray import Ndarray
from taichi.lang.matrix import Vector
from taichi.lang.util import to_taichi_type


def get_field_info(field):
    info = _ti_core.FieldInfo()
    if field is None:
        info.valid = False
        return info
    info.valid = True
    # NDArray & numpy.ndarray
    if isinstance(field, np.ndarray):
        info.field_source = _ti_core.FieldSource.HostMappedPtr
        info.dev_alloc = _ti_core.DeviceAllocation(0, field.ctypes.data)
        info.dtype = to_taichi_type(field.dtype)
        info.num_elements = np.prod(field.shape)
        info.shape = field.shape
        return info
    if isinstance(field, Ndarray):
        info.field_source = _ti_core.FieldSource.TaichiNDarray
        info.dev_alloc = field.arr.get_device_allocation()
        info.dtype = field.dtype
        info.num_elements = np.prod(field.shape) * np.prod(field.element_shape)
        info.shape = field.shape + field.element_shape
        return info
    # SNode
    raise Exception("unsupported data source")


def euler_to_vec(yaw, pitch):
    v = Vector([0.0, 0.0, 0.0])
    v[0] = sin(yaw) * cos(pitch)
    v[1] = sin(pitch)
    v[2] = cos(yaw) * cos(pitch)
    return v


def vec_to_euler(v, eps: float = 1e-6):
    v = v.normalized()
    pitch = asin(v[1])

    cos_pitch = np.sqrt(1 - v[1] * v[1])

    sin_yaw = v[0] / cos_pitch
    cos_yaw = v[2] / cos_pitch

    yaw = np.arctan2(sin_yaw, cos_yaw)

    return yaw, pitch


class GGUINotAvailableException(Exception):
    pass


def check_ggui_availability():
    """Checks if the `GGUI` environment is available."""
    if _ti_core.GGUI_AVAILABLE:
        return

    try:
        # Try identifying the reason
        import taichi  # pylint: disable=import-outside-toplevel

        wheel_tag = try_get_wheel_tag(taichi)
        if platform.system() == "Linux" and wheel_tag and "manylinux2014" in wheel_tag:
            raise GGUINotAvailableException(
                "GGUI is not available since you have installed a restricted version of taichi. "
                "Please see yellow warning messages printed during startup for details."
            )

    except GGUINotAvailableException:
        raise

    except Exception:
        pass

    raise GGUINotAvailableException("GGUI is not available.")
