from taichi.core.util import ti_core as _ti_core

# Real types

float32 = _ti_core.DataType_f32
f32 = float32
float64 = _ti_core.DataType_f64
f64 = float64

real_types = [f32, f64, float]
real_type_ids = [id(t) for t in real_types]

# Integer types

int8 = _ti_core.DataType_i8
i8 = int8
int16 = _ti_core.DataType_i16
i16 = int16
int32 = _ti_core.DataType_i32
i32 = int32
int64 = _ti_core.DataType_i64
i64 = int64

uint8 = _ti_core.DataType_u8
u8 = uint8
uint16 = _ti_core.DataType_u16
u16 = uint16
uint32 = _ti_core.DataType_u32
u32 = uint32
uint64 = _ti_core.DataType_u64
u64 = uint64

integer_types = [i8, i16, i32, i64, u8, u16, u32, u64, int]
integer_type_ids = [id(t) for t in integer_types]

types = real_types + integer_types
type_ids = [id(t) for t in types]

__all__ = [
    'float32',
    'f32',
    'float64',
    'f64',
    'int8',
    'i8',
    'int16',
    'i16',
    'int32',
    'i32',
    'int64',
    'i64',
    'uint8',
    'u8',
    'uint16',
    'u16',
    'uint32',
    'u32',
    'uint64',
    'u64',
]
