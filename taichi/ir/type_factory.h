#pragma once

#include "taichi/ir/type.h"

#include <mutex>

TLANG_NAMESPACE_BEGIN

class TypeFactory {
 public:
  static TypeFactory &get_instance();

  // TODO(type): maybe it makes sense to let each get_X function return X*
  // instead of generic Type*

  Type *get_primitive_type(PrimitiveTypeID id);

  PrimitiveType *get_primitive_int_type(int bits, bool is_signed = true);

  PrimitiveType *get_primitive_real_type(int bits);

  Type *get_tensor_type(std::vector<int> shape, Type *element);

  Type *get_pointer_type(Type *element, bool is_bit_pointer = false);

  Type *get_quant_int_type(int num_bits, bool is_signed, Type *compute_type);

  Type *get_quant_fixed_type(Type *digits_type,
                             Type *compute_type,
                             float64 scale);

  Type *get_quant_float_type(Type *digits_type,
                             Type *exponent_type,
                             Type *compute_type);

  BitStructType *get_bit_struct_type(
      PrimitiveType *physical_type,
      const std::vector<Type *> &member_types,
      const std::vector<int> &member_bit_offsets,
      const std::vector<bool> &member_owns_shared_exponents,
      const std::vector<int> &member_exponents,
      const std::vector<std::vector<int>> &member_exponent_users);

  Type *get_quant_array_type(PrimitiveType *physical_type,
                             Type *element_type,
                             int num_elements);

  static DataType create_vector_or_scalar_type(int width,
                                               DataType element,
                                               bool element_is_pointer = false);

  static DataType create_tensor_type(std::vector<int> shape, DataType element);

 private:
  TypeFactory();

  std::unordered_map<PrimitiveTypeID, std::unique_ptr<Type>> primitive_types_;

  // TODO: use unordered map
  std::map<std::pair<int, Type *>, std::unique_ptr<Type>> vector_types_;

  // TODO: use unordered map
  std::map<std::pair<std::string, Type *>, std::unique_ptr<Type>> tensor_types_;

  // TODO: is_bit_ptr?
  std::map<std::pair<Type *, bool>, std::unique_ptr<Type>> pointer_types_;

  // TODO: use unordered map
  std::map<std::tuple<int, bool, Type *>, std::unique_ptr<Type>>
      quant_int_types_;

  // TODO: use unordered map
  std::map<std::tuple<Type *, Type *, float64>, std::unique_ptr<Type>>
      quant_fixed_types_;

  // TODO: use unordered map
  std::map<std::tuple<Type *, Type *, Type *>, std::unique_ptr<Type>>
      quant_float_types_;

  // TODO: avoid duplication
  std::vector<std::unique_ptr<BitStructType>> bit_struct_types_;

  // TODO: avoid duplication
  std::vector<std::unique_ptr<Type>> quant_array_types_;

  std::mutex mut_;
};

DataType promoted_type(DataType a, DataType b);

TLANG_NAMESPACE_END
