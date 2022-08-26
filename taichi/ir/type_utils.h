#pragma once

#include "taichi/ir/type.h"
#include "taichi/ir/type_factory.h"

namespace taichi {
namespace lang {

std::vector<int> data_type_shape(DataType t);

TI_DLL_EXPORT std::string data_type_name(DataType t);

TI_DLL_EXPORT int data_type_size(DataType t);

TI_DLL_EXPORT std::string data_type_format(DataType dt);

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
  return dt->is<QuantIntType>() || dt->is<QuantFixedType>() ||
         dt->is<QuantFloatType>();
}

inline bool is_real(DataType dt) {
  return dt->is_primitive(PrimitiveTypeID::f16) ||
         dt->is_primitive(PrimitiveTypeID::f32) ||
         dt->is_primitive(PrimitiveTypeID::f64) || dt->is<QuantFixedType>() ||
         dt->is<QuantFloatType>();
}

inline bool is_integral(DataType dt) {
  return dt->is_primitive(PrimitiveTypeID::i8) ||
         dt->is_primitive(PrimitiveTypeID::i16) ||
         dt->is_primitive(PrimitiveTypeID::i32) ||
         dt->is_primitive(PrimitiveTypeID::i64) ||
         dt->is_primitive(PrimitiveTypeID::u8) ||
         dt->is_primitive(PrimitiveTypeID::u16) ||
         dt->is_primitive(PrimitiveTypeID::u32) ||
         dt->is_primitive(PrimitiveTypeID::u64) || dt->is<QuantIntType>();
}

inline bool is_signed(DataType dt) {
  // Shall we return false if is_integral returns false?
  TI_ASSERT(is_integral(dt));
  if (auto t = dt->cast<QuantIntType>())
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

class BitStructTypeBuilder {
 public:
  explicit BitStructTypeBuilder(int max_num_bits) {
    physical_type_ =
        TypeFactory::get_instance().get_primitive_int_type(max_num_bits);
  }

  int add_member(Type *member_type) {
    if (auto qflt = member_type->cast<QuantFloatType>()) {
      auto exponent_type = qflt->get_exponent_type();
      auto exponent_id = -1;
      if (is_placing_shared_exponent_ && current_shared_exponent_ != -1) {
        // Reuse existing exponent
        TI_ASSERT_INFO(member_types_[current_shared_exponent_] == exponent_type,
                       "QuantFloatTypes with shared exponents must have "
                       "exactly the same exponent type.");
        exponent_id = current_shared_exponent_;
      } else {
        exponent_id = add_member_impl(exponent_type);
        if (is_placing_shared_exponent_) {
          current_shared_exponent_ = exponent_id;
        }
      }
      auto digits_id = add_member_impl(member_type);
      member_exponents_[digits_id] = exponent_id;
      member_exponent_users_[exponent_id].push_back(digits_id);
      return digits_id;
    }
    return add_member_impl(member_type);
  }

  void begin_placing_shared_exponent() {
    TI_ASSERT(!is_placing_shared_exponent_);
    TI_ASSERT(current_shared_exponent_ == -1);
    is_placing_shared_exponent_ = true;
  }

  void end_placing_shared_exponent() {
    TI_ASSERT(is_placing_shared_exponent_);
    TI_ASSERT(current_shared_exponent_ != -1);
    current_shared_exponent_ = -1;
    is_placing_shared_exponent_ = false;
  }

  BitStructType *build() const {
    return TypeFactory::get_instance().get_bit_struct_type(
        physical_type_, member_types_, member_bit_offsets_, member_exponents_,
        member_exponent_users_);
  }

 private:
  int add_member_impl(Type *member_type) {
    int old_num_members = member_types_.size();
    member_types_.push_back(member_type);
    member_bit_offsets_.push_back(member_total_bits_);
    member_exponents_.push_back(-1);
    member_exponent_users_.push_back({});
    QuantIntType *member_qit = nullptr;
    if (auto qit = member_type->cast<QuantIntType>()) {
      member_qit = qit;
    } else if (auto qfxt = member_type->cast<QuantFixedType>()) {
      member_qit = qfxt->get_digits_type()->as<QuantIntType>();
    } else if (auto qflt = member_type->cast<QuantFloatType>()) {
      member_qit = qflt->get_digits_type()->as<QuantIntType>();
    } else {
      TI_ERROR("Only a QuantType can be a member of a BitStructType.");
    }
    member_total_bits_ += member_qit->get_num_bits();
    auto physical_bits = data_type_bits(physical_type_);
    TI_ERROR_IF(member_total_bits_ > physical_bits,
                "BitStructType overflows: {} bits used out of {}.",
                member_total_bits_, physical_bits);
    return old_num_members;
  }

  PrimitiveType *physical_type_{nullptr};
  std::vector<Type *> member_types_;
  std::vector<int> member_bit_offsets_;
  int member_total_bits_{0};
  std::vector<int> member_exponents_;
  std::vector<std::vector<int>> member_exponent_users_;
  bool is_placing_shared_exponent_{false};
  int current_shared_exponent_{-1};
};

}  // namespace lang
}  // namespace taichi
