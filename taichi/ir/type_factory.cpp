#include "taichi/ir/type_factory.h"

#include "taichi/ir/type_utils.h"

namespace taichi::lang {

TypeFactory &TypeFactory::get_instance() {
  static TypeFactory *type_factory = new TypeFactory;
  return *type_factory;
}

TypeFactory::TypeFactory() {
}

Type *TypeFactory::get_primitive_type(PrimitiveTypeID id) {
  std::lock_guard<std::mutex> _(primitive_mut_);

  if (primitive_types_.find(id) == primitive_types_.end()) {
    primitive_types_[id] = std::make_unique<PrimitiveType>(id);
  }

  return primitive_types_[id].get();
}

Type *TypeFactory::get_tensor_type(std::vector<int> shape, Type *element) {
  std::lock_guard<std::mutex> _(tensor_mut_);

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

Type *TypeFactory::get_struct_type(const std::vector<StructMember> &elements) {
  std::lock_guard<std::mutex> _(struct_mut_);

  if (struct_types_.find(elements) == struct_types_.end()) {
    for (const auto &[type, name, __] : elements) {
      TI_ASSERT_INFO(type->is<PrimitiveType>() || type->is<TensorType>() ||
                         type->is<StructType>() || type->is<PointerType>(),
                     "Unsupported struct element type for element " + name +
                         ": " + type->to_string());
    }
    struct_types_[elements] = std::make_unique<StructType>(elements);
  }
  return struct_types_[elements].get();
}

Type *TypeFactory::get_pointer_type(Type *element, bool is_bit_pointer) {
  std::lock_guard<std::mutex> _(pointer_mut_);

  auto key = std::make_pair(element, is_bit_pointer);
  if (pointer_types_.find(key) == pointer_types_.end()) {
    pointer_types_[key] =
        std::make_unique<PointerType>(element, is_bit_pointer);
  }
  return pointer_types_[key].get();
}

Type *TypeFactory::get_quant_int_type(int num_bits,
                                      bool is_signed,
                                      Type *compute_type) {
  std::lock_guard<std::mutex> _(quant_int_mut_);

  auto key = std::make_tuple(num_bits, is_signed, compute_type);
  if (quant_int_types_.find(key) == quant_int_types_.end()) {
    quant_int_types_[key] =
        std::make_unique<QuantIntType>(num_bits, is_signed, compute_type);
  }
  return quant_int_types_[key].get();
}

Type *TypeFactory::get_quant_fixed_type(Type *digits_type,
                                        Type *compute_type,
                                        float64 scale) {
  std::lock_guard<std::mutex> _(quant_fixed_mut_);

  auto key = std::make_tuple(digits_type, compute_type, scale);
  if (quant_fixed_types_.find(key) == quant_fixed_types_.end()) {
    quant_fixed_types_[key] =
        std::make_unique<QuantFixedType>(digits_type, compute_type, scale);
  }
  return quant_fixed_types_[key].get();
}

Type *TypeFactory::get_quant_float_type(Type *digits_type,
                                        Type *exponent_type,
                                        Type *compute_type) {
  std::lock_guard<std::mutex> _(quant_float_mut_);

  auto key = std::make_tuple(digits_type, exponent_type, compute_type);
  if (quant_float_types_.find(key) == quant_float_types_.end()) {
    quant_float_types_[key] = std::make_unique<QuantFloatType>(
        digits_type, exponent_type, compute_type);
  }
  return quant_float_types_[key].get();
}

BitStructType *TypeFactory::get_bit_struct_type(
    PrimitiveType *physical_type,
    const std::vector<Type *> &member_types,
    const std::vector<int> &member_bit_offsets,
    const std::vector<int> &member_exponents,
    const std::vector<std::vector<int>> &member_exponent_users) {
  std::lock_guard<std::mutex> _(bit_struct_mut_);

  bit_struct_types_.push_back(std::make_unique<BitStructType>(
      physical_type, member_types, member_bit_offsets, member_exponents,
      member_exponent_users));
  return bit_struct_types_.back().get();
}

Type *TypeFactory::get_quant_array_type(PrimitiveType *physical_type,
                                        Type *element_type,
                                        int num_elements) {
  std::lock_guard<std::mutex> _(quant_array_mut_);

  quant_array_types_.push_back(std::make_unique<QuantArrayType>(
      physical_type, element_type, num_elements));
  return quant_array_types_.back().get();
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

DataType promoted_primitive_type(DataType x, DataType y) {
  if (compare_types(to_primitive_type(x), to_primitive_type(y)))
    return x;
  else
    return y;
}

DataType promoted_type(DataType a, DataType b) {
  if (a->is<TensorType>() || b->is<TensorType>()) {
    TI_ASSERT_INFO(a->is<TensorType>() && b->is<TensorType>(),
                   "a = {}, b = {}, only one of them is a tensor type",
                   a->to_string(), b->to_string());
    auto tensor_ty_a = a->cast<TensorType>();
    auto tensor_ty_b = b->cast<TensorType>();
    auto promoted_dt = promoted_type(tensor_ty_a->get_element_type(),
                                     tensor_ty_b->get_element_type());
    return TypeFactory::create_tensor_type(tensor_ty_a->get_shape(),
                                           promoted_dt);
  } else {
    return promoted_primitive_type(a, b);
  }
};

}  // namespace taichi::lang
