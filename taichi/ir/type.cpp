#include "taichi/ir/type.h"

#include "taichi/ir/type_factory.h"
#include "taichi/ir/type_utils.h"

TLANG_NAMESPACE_BEGIN

// Note: these primitive types should never be freed. They are supposed to live
// together with the process. This is a temporary solution. Later we should
// manage its ownership more systematically.

// This part doesn't look good, but we will remove it soon anyway.
#define PER_TYPE(x)                     \
  DataType PrimitiveType::x = DataType( \
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::x));

#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

DataType::DataType() : ptr_(PrimitiveType::unknown.ptr_) {
}

DataType PrimitiveType::get(PrimitiveTypeID t) {
  if (false) {
  }
#define PER_TYPE(x) else if (t == PrimitiveTypeID::x) return PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  else {
    TI_NOT_IMPLEMENTED
  }
}

std::size_t DataType::hash() const {
  if (auto primitive = ptr_->cast<PrimitiveType>()) {
    return (std::size_t)primitive->type;
  } else if (auto pointer = ptr_->cast<PointerType>()) {
    return 10007 + DataType(pointer->get_pointee_type()).hash();
  } else {
    TI_NOT_IMPLEMENTED
  }
}

bool DataType::is_pointer() const {
  return ptr_->is<PointerType>();
}

void DataType::set_is_pointer(bool is_ptr) {
  if (is_ptr && !ptr_->is<PointerType>()) {
    ptr_ = TypeFactory::get_instance().get_pointer_type(ptr_);
  }
  if (!is_ptr && ptr_->is<PointerType>()) {
    ptr_ = ptr_->cast<PointerType>()->get_pointee_type();
  }
}

DataType DataType::ptr_removed() const {
  auto t = ptr_;
  auto ptr_type = t->cast<PointerType>();
  if (ptr_type) {
    return DataType(ptr_type->get_pointee_type());
  } else {
    return *this;
  }
}

std::string PrimitiveType::to_string() const {
  return data_type_name(DataType(const_cast<PrimitiveType *>(this)));
}

std::string PointerType::to_string() const {
  if (is_bit_pointer_) {
    // "^" for bit-level pointers
    return fmt::format("^{}", pointee_->to_string());
  } else {
    // "*" for C-style byte-level pointers
    return fmt::format("*{}", pointee_->to_string());
  }
}

std::string TensorType::to_string() const {
  std::string s = "[Tensor (";
  for (int i = 0; i < (int)shape_.size(); ++i) {
    s += fmt::format(i == 0 ? "{}" : ", {}", shape_[i]);
  }
  s += fmt::format(") {}]", element_->to_string());
  return s;
}

int Type::vector_width() const {
  return 1;  // TODO: CPU vectorization
}

bool Type::is_primitive(PrimitiveTypeID type) const {
  if (auto p = cast<PrimitiveType>()) {
    return p->type == type;
  } else {
    return false;
  }
}

std::string CustomIntType::to_string() const {
  return fmt::format("c{}{}", is_signed_ ? 'i' : 'u', num_bits_);
}

CustomIntType::CustomIntType(int num_bits,
                             bool is_signed,
                             Type *compute_type,
                             Type *physical_type)
    : compute_type_(compute_type),
      physical_type_(physical_type),
      num_bits_(num_bits),
      is_signed_(is_signed) {
  if (compute_type == nullptr) {
    auto type_id = is_signed ? PrimitiveTypeID::i32 : PrimitiveTypeID::u32;
    this->compute_type_ =
        TypeFactory::get_instance().get_primitive_type(type_id);
  }
}

CustomFloatType::CustomFloatType(Type *digits_type,
                                 Type *exponent_type,
                                 Type *compute_type,
                                 float64 scale)
    : digits_type_(digits_type),
      exponent_type_(exponent_type),
      compute_type_(compute_type),
      scale_(scale) {
  TI_ASSERT(digits_type->is<CustomIntType>());
  TI_ASSERT(compute_type->is<PrimitiveType>());
  TI_ASSERT(is_real(compute_type->as<PrimitiveType>()));

  if (exponent_type_) {
    // We only support f32 as compute type when when using exponents
    TI_ASSERT(compute_type_->is_primitive(PrimitiveTypeID::f32));
    // Exponent must be unsigned custom int
    TI_ASSERT(exponent_type->is<CustomIntType>());
    TI_ASSERT(exponent_type->as<CustomIntType>()->get_num_bits() <= 8);
    TI_ASSERT(exponent_type->as<CustomIntType>()->get_is_signed() == false);
    TI_ASSERT(get_digit_bits() <= 23);
  }
}

std::string CustomFloatType::to_string() const {
  std::string e, s;
  if (exponent_type_)
    e = fmt::format(" e={}", exponent_type_->to_string());
  if (scale_ != 1)
    s = fmt::format(" s={}", scale_);
  return fmt::format("cf(d={}{} c={}{})", digits_type_->to_string(), e,
                     compute_type_->to_string(), s);
}

int CustomFloatType::get_exponent_conversion_offset() const {
  // Note that f32 has exponent offset -127
  return 127 -
         (1 << (exponent_type_->as<CustomIntType>()->get_num_bits() - 1)) + 1;
}

int CustomFloatType::get_digit_bits() const {
  return digits_type_->as<CustomIntType>()->get_num_bits() -
         (int)get_is_signed();
}

bool CustomFloatType::get_is_signed() const {
  return digits_type_->as<CustomIntType>()->get_is_signed();
}

BitStructType::BitStructType(PrimitiveType *physical_type,
                             std::vector<Type *> member_types,
                             std::vector<int> member_bit_offsets)
    : physical_type_(physical_type),
      member_types_(member_types),
      member_bit_offsets_(member_bit_offsets) {
  TI_ASSERT(member_types_.size() == member_bit_offsets_.size());
  int physical_type_bits = data_type_bits(physical_type);
  for (auto i = 0; i < member_types_.size(); ++i) {
    CustomIntType *component_cit = nullptr;
    if (auto cit = member_types_[i]->cast<CustomIntType>()) {
      component_cit = cit;
    } else if (auto cft = member_types_[i]->cast<CustomFloatType>()) {
      component_cit = cft->get_digits_type()->as<CustomIntType>();
    } else {
      TI_NOT_IMPLEMENTED
    }
    auto bits_end = component_cit->get_num_bits() + member_bit_offsets_[i];
    TI_ASSERT(physical_type_bits >= bits_end)
  }
}

std::string BitStructType::to_string() const {
  std::string str = "bs(";
  int num_members = (int)member_bit_offsets_.size();
  for (int i = 0; i < num_members; i++) {
    str += fmt::format("{}@{}", member_types_[i]->to_string(),
                       member_bit_offsets_[i]);
    if (i + 1 < num_members) {
      str += ", ";
    }
  }
  return str + ")";
}

std::string BitArrayType::to_string() const {
  return fmt::format("ba({}x{})", element_type_->to_string(), num_elements_);
}

std::string TypedConstant::stringify() const {
  // TODO: remove the line below after type system upgrade.
  auto dt = this->dt.ptr_removed();
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return fmt::format("{}", val_f32);
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return fmt::format("{}", val_i32);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return fmt::format("{}", val_i64);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return fmt::format("{}", val_f64);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return fmt::format("{}", val_i8);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return fmt::format("{}", val_i16);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return fmt::format("{}", val_u8);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return fmt::format("{}", val_u16);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return fmt::format("{}", val_u32);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return fmt::format("{}", val_u64);
  } else {
    TI_P(data_type_name(dt));
    TI_NOT_IMPLEMENTED
    return "";
  }
}

bool TypedConstant::equal_type_and_value(const TypedConstant &o) const {
  if (dt != o.dt)
    return false;
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return val_f32 == o.val_f32;
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return val_i32 == o.val_i32;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return val_i64 == o.val_i64;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return val_f64 == o.val_f64;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return val_i8 == o.val_i8;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return val_i16 == o.val_i16;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return val_u8 == o.val_u8;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return val_u16 == o.val_u16;
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return val_u32 == o.val_u32;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return val_u64 == o.val_u64;
  } else {
    TI_NOT_IMPLEMENTED
    return false;
  }
}

int32 &TypedConstant::val_int32() {
  TI_ASSERT(get_data_type<int32>() == dt);
  return val_i32;
}

float32 &TypedConstant::val_float32() {
  TI_ASSERT(get_data_type<float32>() == dt);
  return val_f32;
}

int64 &TypedConstant::val_int64() {
  TI_ASSERT(get_data_type<int64>() == dt);
  return val_i64;
}

float64 &TypedConstant::val_float64() {
  TI_ASSERT(get_data_type<float64>() == dt);
  return val_f64;
}

int8 &TypedConstant::val_int8() {
  TI_ASSERT(get_data_type<int8>() == dt);
  return val_i8;
}

int16 &TypedConstant::val_int16() {
  TI_ASSERT(get_data_type<int16>() == dt);
  return val_i16;
}

uint8 &TypedConstant::val_uint8() {
  TI_ASSERT(get_data_type<uint8>() == dt);
  return val_u8;
}

uint16 &TypedConstant::val_uint16() {
  TI_ASSERT(get_data_type<uint16>() == dt);
  return val_u16;
}

uint32 &TypedConstant::val_uint32() {
  TI_ASSERT(get_data_type<uint32>() == dt);
  return val_u32;
}

uint64 &TypedConstant::val_uint64() {
  TI_ASSERT(get_data_type<uint64>() == dt);
  return val_u64;
}

int64 TypedConstant::val_int() const {
  TI_ASSERT(is_signed(dt));
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return val_i32;
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return val_i64;
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return val_i8;
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return val_i16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

uint64 TypedConstant::val_uint() const {
  TI_ASSERT(is_unsigned(dt));
  if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return val_u32;
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return val_u64;
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return val_u8;
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return val_u16;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 TypedConstant::val_float() const {
  TI_ASSERT(is_real(dt));
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return val_f32;
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return val_f64;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

int64 TypedConstant::val_as_int64() const {
  if (is_real(dt)) {
    TI_ERROR("Cannot cast floating point type {} to int64.", dt->to_string());
  } else if (is_signed(dt)) {
    return val_int();
  } else if (is_unsigned(dt)) {
    return val_uint();
  } else {
    TI_NOT_IMPLEMENTED
  }
}

float64 TypedConstant::val_cast_to_float64() const {
  if (is_real(dt))
    return val_float();
  else if (is_signed(dt))
    return val_int();
  else if (is_unsigned(dt))
    return val_uint();
  else {
    TI_NOT_IMPLEMENTED
  }
}

TLANG_NAMESPACE_END
