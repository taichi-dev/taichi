#include "taichi/ir/type_factory.h"

#include "taichi/ir/type_utils.h"

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

Type *TypeFactory::get_tensor_type(std::vector<int> shape, Type *element) {
  auto encode = [](const std::vector<int> &shape) -> std::string {
    std::string s;
    for (int i = 0; i < (int)shape.size(); ++i)
      s += fmt::format(i == 0 ? "{}" : "_{}", std::to_string(shape[i]));
    return s;
  };
  auto key = std::make_pair(encode(shape), element);
  if (tensor_types_.find(key) == tensor_types_.end()) {
    tensor_types_[key] = std::make_unique<TensorType>(shape, element);
  }
  return tensor_types_[key].get();
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
                                       Type *compute_type) {
  auto key = std::make_tuple(num_bits, is_signed, compute_type);
  if (custom_int_types_.find(key) == custom_int_types_.end()) {
    custom_int_types_[key] =
        std::make_unique<CustomIntType>(num_bits, is_signed, compute_type);
  }
  return custom_int_types_[key].get();
}

Type *TypeFactory::get_custom_float_type(Type *digits_type,
                                         Type *exponent_type,
                                         Type *compute_type,
                                         float64 scale) {
  auto key = std::make_tuple(digits_type, exponent_type, compute_type, scale);
  if (custom_float_types_.find(key) == custom_float_types_.end()) {
    custom_float_types_[key] = std::make_unique<CustomFloatType>(
        digits_type, exponent_type, compute_type, scale);
  }
  return custom_float_types_[key].get();
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
    int_type = to_unsigned(DataType(int_type));
  }
  return int_type->cast<PrimitiveType>();
}

PrimitiveType *TypeFactory::get_primitive_real_type(int bits) {
  Type *real_type;
  if (bits == 16) {
    real_type = get_primitive_type(PrimitiveTypeID::f16);
  } else if (bits == 32) {
    real_type = get_primitive_type(PrimitiveTypeID::f32);
  } else if (bits == 64) {
    real_type = get_primitive_type(PrimitiveTypeID::f64);
  } else {
    TI_ERROR("No primitive real type has {} bits", bits);
  }
  return real_type->cast<PrimitiveType>();
}

DataType TypeFactory::create_vector_or_scalar_type(int width,
                                                   DataType element,
                                                   bool element_is_pointer) {
  TI_ASSERT(width == 1);
  if (element_is_pointer) {
    return TypeFactory::get_instance().get_pointer_type(element);
  } else {
    return element;
  }
}

DataType TypeFactory::create_tensor_type(std::vector<int> shape,
                                         DataType element) {
  return TypeFactory::get_instance().get_tensor_type(shape, element);
}

namespace {
static bool compare_types(DataType x, DataType y) {
  // Is the first type "bigger" than the second type?
  if (is_real(x) != is_real(y)) {
    // One is real, the other is integral.
    // real > integral
    return is_real(x);
  } else {
    if (is_real(x) && is_real(y)) {
      // Both are real
      return data_type_bits(x) > data_type_bits(y);
    } else {
      // Both are integral
      auto x_bits = data_type_bits(x);
      auto y_bits = data_type_bits(y);
      if (x_bits != y_bits) {
        return x_bits > y_bits;
      } else {
        // Same number of bits. Unsigned > signed
        auto x_unsigned = !is_signed(x);
        auto y_unsigned = !is_signed(y);
        return x_unsigned > y_unsigned;
      }
    }
  }
}

static DataType to_primitive_type(DataType d) {
  if (d->is<PointerType>()) {
    d = d->as<PointerType>()->get_pointee_type();
    TI_WARN("promoted_type got a pointer input.");
  }

  if (d->is<TensorType>()) {
    d = d->as<TensorType>()->get_element_type();
    TI_WARN("promoted_type got a tensor input.");
  }

  auto primitive = d->cast<PrimitiveType>();
  TI_ASSERT_INFO(primitive, "Failed to get primitive type from {}",
                 d->to_string());
  return primitive;
};
}  // namespace

DataType promoted_type(DataType x, DataType y) {
  if (compare_types(to_primitive_type(x), to_primitive_type(y)))
    return x;
  else
    return y;
}

TLANG_NAMESPACE_END
