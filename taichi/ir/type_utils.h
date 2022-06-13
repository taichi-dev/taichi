#pragma once

#include "taichi/ir/type.h"

namespace taichi {
namespace lang {

std::string data_type_name(DataType t);

std::string data_type_format(DataType dt);

int data_type_size(DataType t);

inline int data_type_bits(DataType t) {
  return data_type_size(t) * 8;
}

template <typename T>
inline DataType get_data_type() {
  if (std::is_same<T, float32>()) {
    return PrimitiveType::f32;
  } else if (std::is_same<T, float64>()) {
    return PrimitiveType::f64;
  } else if (std::is_same<T, bool>()) {
    return PrimitiveType::u1;
  } else if (std::is_same<T, int8>()) {
    return PrimitiveType::i8;
  } else if (std::is_same<T, int16>()) {
    return PrimitiveType::i16;
  } else if (std::is_same<T, int32>()) {
    return PrimitiveType::i32;
  } else if (std::is_same<T, int64>()) {
    return PrimitiveType::i64;
  } else if (std::is_same<T, uint8>()) {
    return PrimitiveType::u8;
  } else if (std::is_same<T, uint16>()) {
    return PrimitiveType::u16;
  } else if (std::is_same<T, uint32>()) {
    return PrimitiveType::u32;
  } else if (std::is_same<T, uint64>()) {
    return PrimitiveType::u64;
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

template <typename T>
inline PrimitiveTypeID get_primitive_data_type() {
  if (std::is_same<T, float32>()) {
    return PrimitiveTypeID::f32;
  } else if (std::is_same<T, float64>()) {
    return PrimitiveTypeID::f64;
  } else if (std::is_same<T, bool>()) {
    return PrimitiveTypeID::u1;
  } else if (std::is_same<T, int8>()) {
    return PrimitiveTypeID::i8;
  } else if (std::is_same<T, int16>()) {
    return PrimitiveTypeID::i16;
  } else if (std::is_same<T, int32>()) {
    return PrimitiveTypeID::i32;
  } else if (std::is_same<T, int64>()) {
    return PrimitiveTypeID::i64;
  } else if (std::is_same<T, uint8>()) {
    return PrimitiveTypeID::u8;
  } else if (std::is_same<T, uint16>()) {
    return PrimitiveTypeID::u16;
  } else if (std::is_same<T, uint32>()) {
    return PrimitiveTypeID::u32;
  } else if (std::is_same<T, uint64>()) {
    return PrimitiveTypeID::u64;
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

inline bool is_quant(DataType dt) {
  return dt->is<CustomIntType>() || dt->is<CustomFloatType>();
}

inline bool is_real(DataType dt) {
  return dt->is_primitive(PrimitiveTypeID::f16) ||
         dt->is_primitive(PrimitiveTypeID::f32) ||
         dt->is_primitive(PrimitiveTypeID::f64) || dt->is<CustomFloatType>();
}

inline bool is_integral(DataType dt) {
  return dt->is_primitive(PrimitiveTypeID::i8) ||
         dt->is_primitive(PrimitiveTypeID::i16) ||
         dt->is_primitive(PrimitiveTypeID::i32) ||
         dt->is_primitive(PrimitiveTypeID::i64) ||
         dt->is_primitive(PrimitiveTypeID::u8) ||
         dt->is_primitive(PrimitiveTypeID::u16) ||
         dt->is_primitive(PrimitiveTypeID::u32) ||
         dt->is_primitive(PrimitiveTypeID::u64) || dt->is<CustomIntType>();
}

inline bool is_signed(DataType dt) {
  // Shall we return false if is_integral returns false?
  TI_ASSERT(is_integral(dt));
  if (auto t = dt->cast<CustomIntType>())
    return t->get_is_signed();
  return dt->is_primitive(PrimitiveTypeID::i8) ||
         dt->is_primitive(PrimitiveTypeID::i16) ||
         dt->is_primitive(PrimitiveTypeID::i32) ||
         dt->is_primitive(PrimitiveTypeID::i64);
}

inline bool is_unsigned(DataType dt) {
  TI_ASSERT(is_integral(dt));
  return !is_signed(dt);
}

inline DataType to_unsigned(DataType dt) {
  TI_ASSERT(is_signed(dt));
  if (dt->is_primitive(PrimitiveTypeID::i8))
    return PrimitiveType::u8;
  else if (dt->is_primitive(PrimitiveTypeID::i16))
    return PrimitiveType::u16;
  else if (dt->is_primitive(PrimitiveTypeID::i32))
    return PrimitiveType::u32;
  else if (dt->is_primitive(PrimitiveTypeID::i64))
    return PrimitiveType::u64;
  else
    return PrimitiveType::unknown;
}

inline bool needs_grad(DataType dt) {
  return is_real(dt);
}

inline TypedConstant get_max_value(DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return {dt, std::numeric_limits<int8>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return {dt, std::numeric_limits<int16>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return {dt, std::numeric_limits<int32>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return {dt, std::numeric_limits<int64>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return {dt, std::numeric_limits<uint8>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return {dt, std::numeric_limits<uint16>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return {dt, std::numeric_limits<uint32>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return {dt, std::numeric_limits<uint64>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return {dt, std::numeric_limits<float32>::max()};
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return {dt, std::numeric_limits<float64>::max()};
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

inline TypedConstant get_min_value(DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return {dt, std::numeric_limits<int8>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return {dt, std::numeric_limits<int16>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return {dt, std::numeric_limits<int32>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return {dt, std::numeric_limits<int64>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return {dt, std::numeric_limits<uint8>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return {dt, std::numeric_limits<uint16>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return {dt, std::numeric_limits<uint32>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return {dt, std::numeric_limits<uint64>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return {dt, std::numeric_limits<float32>::min()};
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return {dt, std::numeric_limits<float64>::min()};
  } else {
    TI_NOT_IMPLEMENTED;
  }
}

}  // namespace lang
}  // namespace taichi
