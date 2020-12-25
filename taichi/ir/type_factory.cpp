#include "taichi/ir/type_factory.h"

TLANG_NAMESPACE_BEGIN

TypeFactory &TypeFactory::get_instance() {
  static TypeFactory *type_factory = new TypeFactory;
  return *type_factory;
}

TypeFactory::TypeFactory() {
}

Type *TypeFactory::get_primitive_type(PrimitiveTypeID id) {
  std::lock_guard<std::mutex> _(mut_);

  if (primitive_types_.find(id) == primitive_types_.end()) {
    primitive_types_[id] = std::make_unique<PrimitiveType>(id);
  }

  return primitive_types_[id].get();
}

Type *TypeFactory::get_vector_type(int num_elements, Type *element) {
  auto key = std::make_pair(num_elements, element);
  if (vector_types_.find(key) == vector_types_.end()) {
    vector_types_[key] = std::make_unique<VectorType>(num_elements, element);
  }
  return vector_types_[key].get();
}

Type *TypeFactory::get_pointer_type(Type *element, bool is_bit_pointer) {
  auto key = std::make_pair(element, is_bit_pointer);
  if (pointer_types_.find(key) == pointer_types_.end()) {
    pointer_types_[key] =
        std::make_unique<PointerType>(element, is_bit_pointer);
  }
  return pointer_types_[key].get();
}

Type *TypeFactory::get_custom_int_type(int num_bits,
                                       bool is_signed,
                                       int compute_type_bits) {
  auto key = std::make_tuple(compute_type_bits, num_bits, is_signed);
  if (custom_int_types.find(key) == custom_int_types.end()) {
    custom_int_types[key] = std::make_unique<CustomIntType>(
        num_bits, is_signed,
        get_primitive_int_type(compute_type_bits, is_signed));
  }
  return custom_int_types[key].get();
}

Type *TypeFactory::get_custom_float_type(Type *digits_type,
                                         Type *exponent_type,
                                         Type *compute_type,
                                         float64 scale) {
  auto key = std::make_tuple(digits_type, exponent_type, compute_type, scale);
  if (custom_float_types.find(key) == custom_float_types.end()) {
    custom_float_types[key] = std::make_unique<CustomFloatType>(
        digits_type, exponent_type, compute_type, scale);
  }
  return custom_float_types[key].get();
}

Type *TypeFactory::get_bit_struct_type(PrimitiveType *physical_type,
                                       std::vector<Type *> member_types,
                                       std::vector<int> member_bit_offsets) {
  bit_struct_types_.push_back(std::make_unique<BitStructType>(
      physical_type, member_types, member_bit_offsets));
  return bit_struct_types_.back().get();
}

Type *TypeFactory::get_bit_array_type(PrimitiveType *physical_type,
                                      Type *element_type,
                                      int num_elements) {
  bit_array_types_.push_back(std::make_unique<BitArrayType>(
      physical_type, element_type, num_elements));
  return bit_array_types_.back().get();
}

PrimitiveType *TypeFactory::get_primitive_int_type(int bits, bool is_signed) {
  Type *int_type;
  if (bits == 8) {
    int_type = get_primitive_type(PrimitiveTypeID::i8);
  } else if (bits == 16) {
    int_type = get_primitive_type(PrimitiveTypeID::i16);
  } else if (bits == 32) {
    int_type = get_primitive_type(PrimitiveTypeID::i32);
  } else if (bits == 64) {
    int_type = get_primitive_type(PrimitiveTypeID::i64);
  } else {
    TI_ERROR("No primitive int type has {} bits", bits);
  }
  if (!is_signed) {
    int_type = to_unsigned(DataType(int_type)).get_ptr();
  }
  return int_type->cast<PrimitiveType>();
}

DataType TypeFactory::create_vector_or_scalar_type(int width,
                                                   DataType element,
                                                   bool element_is_pointer) {
  TI_ASSERT(width == 1);
  if (element_is_pointer) {
    return TypeFactory::get_instance().get_pointer_type(element.get_ptr());
  } else {
    return element;
  }
}

TLANG_NAMESPACE_END
