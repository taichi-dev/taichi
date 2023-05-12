from taichi._lib import core as ti_python_core

# ========================================
# real types

# ----------------------------------------

float16 = ti_python_core.DataType_f16
"""16-bit precision floating point data type.
"""

# ----------------------------------------

f16 = float16
"""Alias for :const:`~taichi.types.primitive_types.float16`
"""

# ----------------------------------------

float32 = ti_python_core.DataType_f32
"""32-bit single precision floating point data type.
"""

# ----------------------------------------

f32 = float32
"""Alias for :const:`~taichi.types.primitive_types.float32`
"""

# ----------------------------------------

float64 = ti_python_core.DataType_f64
"""64-bit double precision floating point data type.
"""

# ----------------------------------------

f64 = float64
"""Alias for :const:`~taichi.types.primitive_types.float64`
"""
# ----------------------------------------

# ========================================
# Integer types

# ----------------------------------------

int8 = ti_python_core.DataType_i8
"""8-bit signed integer data type.
"""

# ----------------------------------------

i8 = int8
"""Alias for :const:`~taichi.types.primitive_types.int8`
"""

# ----------------------------------------

int16 = ti_python_core.DataType_i16
"""16-bit signed integer data type.
"""

# ----------------------------------------

i16 = int16
"""Alias for :const:`~taichi.types.primitive_types.int16`
"""

# ----------------------------------------

int32 = ti_python_core.DataType_i32
"""32-bit signed integer data type.
"""

# ----------------------------------------

i32 = int32
"""Alias for :const:`~taichi.types.primitive_types.int32`
"""

# ----------------------------------------

int64 = ti_python_core.DataType_i64
"""64-bit signed integer data type.
"""

# ----------------------------------------

i64 = int64
"""Alias for :const:`~taichi.types.primitive_types.int64`
"""

# ----------------------------------------

uint8 = ti_python_core.DataType_u8
"""8-bit unsigned integer data type.
"""

# ----------------------------------------

uint1 = ti_python_core.DataType_u1
"""1-bit unsigned integer data type. Same as booleans.
"""

# ----------------------------------------

u1 = uint1
"""Alias for :const:`~taichi.types.primitive_types.uint1`
"""

# ----------------------------------------

u8 = uint8
"""Alias for :const:`~taichi.types.primitive_types.uint8`
"""

# ----------------------------------------

uint16 = ti_python_core.DataType_u16
"""16-bit unsigned integer data type.
"""

# ----------------------------------------

u16 = uint16
"""Alias for :const:`~taichi.types.primitive_types.uint16`
"""

# ----------------------------------------

uint32 = ti_python_core.DataType_u32
"""32-bit unsigned integer data type.
"""

# ----------------------------------------

u32 = uint32
"""Alias for :const:`~taichi.types.primitive_types.uint32`
"""

# ----------------------------------------

uint64 = ti_python_core.DataType_u64
"""64-bit unsigned integer data type.
"""

# ----------------------------------------

u64 = uint64
"""Alias for :const:`~taichi.types.primitive_types.uint64`
"""

# ----------------------------------------


class RefType:
    def __init__(self, tp):
        self.tp = tp


def ref(tp):
    return RefType(tp)


real_types = [f16, f32, f64, float]
real_type_ids = [id(t) for t in real_types]

integer_types = [i8, i16, i32, i64, u1, u8, u16, u32, u64, int, bool]
integer_type_ids = [id(t) for t in integer_types]

all_types = real_types + integer_types
type_ids = [id(t) for t in all_types]

__all__ = [
    "float32",
    "f32",
    "float64",
    "f64",
    "float16",
    "f16",
    "int8",
    "i8",
    "int16",
    "i16",
    "int32",
    "i32",
    "int64",
    "i64",
    "uint1",
    "u1",
    "uint8",
    "u8",
    "uint16",
    "u16",
    "uint32",
    "u32",
    "uint64",
    "u64",
    "ref",
]
