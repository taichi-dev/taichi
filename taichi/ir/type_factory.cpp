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
class TypePromotionMapping {
 public:
  TypePromotionMapping() {
#define TRY_SECOND(x, y)                                   \
  mapping[std::make_pair(get_primitive_data_type<x>(),     \
                         get_primitive_data_type<y>())] =  \
      get_primitive_data_type<decltype(std::declval<x>() + \
                                       std::declval<y>())>();
#define TRY_FIRST(x)      \
  TRY_SECOND(x, float32); \
  TRY_SECOND(x, float64); \
  TRY_SECOND(x, int8);    \
  TRY_SECOND(x, int16);   \
  TRY_SECOND(x, int32);   \
  TRY_SECOND(x, int64);   \
  TRY_SECOND(x, uint8);   \
  TRY_SECOND(x, uint16);  \
  TRY_SECOND(x, uint32);  \
  TRY_SECOND(x, uint64);

    TRY_FIRST(float32);
    TRY_FIRST(float64);
    TRY_FIRST(int8);
    TRY_FIRST(int16);
    TRY_FIRST(int32);
    TRY_FIRST(int64);
    TRY_FIRST(uint8);
    TRY_FIRST(uint16);
    TRY_FIRST(uint32);
    TRY_FIRST(uint64);
  }

  bool compare(DataType x, DataType y) {
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

  DataType promoted_type(DataType x, DataType y) {
    if (compare(x, y))
      return x;
    else
      return y;
  }

  DataType query(DataType x, DataType y) {
    for (auto &inputs: mapping) {
      auto type_x = TypeFactory::get_instance().get_primitive_type(inputs.first.first);
      auto type_y = TypeFactory::get_instance().get_primitive_type(inputs.first.second);
      auto x_name = data_type_name(type_x);
      auto y_name = data_type_name(type_y);
      auto old_name = data_type_name(TypeFactory::get_instance().get_primitive_type(inputs.second));
      auto new_name = data_type_name(promoted_type(type_x, type_y));

      if (old_name == new_name) {
        fmt::print("{} + {} = {}\n", x_name, y_name, old_name);
      } else {
        fmt::print("{} + {} = {} -> {}\n", x_name, y_name, old_name, new_name);
      }
      // TI_ASSERT(inputs.second == inputs.first.second || inputs.second == inputs.first.first);
    }
    exit(0);

    auto primitive =
        mapping[std::make_pair(to_primitive_type(x), to_primitive_type(y))];

    /*
    */
    return TypeFactory::get_instance().get_primitive_type(primitive);
  }

 private:
  std::map<std::pair<PrimitiveTypeID, PrimitiveTypeID>, PrimitiveTypeID>
      mapping;
  static PrimitiveTypeID to_primitive_type(DataType d) {
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
    return primitive->type;
  };
};
// TODO(#2196): Stop using global variables.
TypePromotionMapping type_promotion_mapping;
}  // namespace

DataType promoted_type(DataType a, DataType b) {
  return type_promotion_mapping.query(a, b);
}

TLANG_NAMESPACE_END
