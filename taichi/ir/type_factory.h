#pragma once

#include "taichi/ir/type.h"
#include "taichi/util/hash.h"

#include <mutex>

namespace taichi::lang {

class TypeFactory {
 public:
  static TypeFactory &get_instance();

  // TODO(type): maybe it makes sense to let each get_X function return X*
  // instead of generic Type*

  const Type *get_primitive_type(PrimitiveTypeID id);

  const PrimitiveType *get_primitive_int_type(int bits, bool is_signed = true);

  const PrimitiveType *get_primitive_real_type(int bits);

  const Type *get_tensor_type(std::vector<int> shape, Type *element);

  const Type *get_struct_type(const std::vector<StructMember> &elements,
                              const std::string &layout = "none");

  const Type *get_pointer_type(const Type *element,
                               bool is_bit_pointer = false);

  const Type *get_quant_int_type(int num_bits,
                                 bool is_signed,
                                 const Type *compute_type);

  const Type *get_quant_fixed_type(const Type *digits_type,
                                   const Type *compute_type,
                                   float64 scale);

  const Type *get_quant_float_type(const Type *digits_type,
                                   const Type *exponent_type,
                                   const Type *compute_type);

  BitStructType *get_bit_struct_type(
      const PrimitiveType *physical_type,
      const std::vector<const Type *> &member_types,
      const std::vector<int> &member_bit_offsets,
      const std::vector<int> &member_exponents,
      const std::vector<std::vector<int>> &member_exponent_users);

  const Type *get_quant_array_type(const PrimitiveType *physical_type,
                                   const Type *element_type,
                                   int num_elements);

  static DataType create_tensor_type(std::vector<int> shape, DataType element);

 private:
  TypeFactory();

  std::unordered_map<PrimitiveTypeID, std::unique_ptr<Type>> primitive_types_;
  std::mutex primitive_mut_;

  std::unordered_map<std::pair<std::string, const Type *>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::pair<std::string, const Type *>>>
      tensor_types_;
  std::mutex tensor_mut_;

  std::unordered_map<
      std::pair<std::vector<StructMember>, std::string>,
      std::unique_ptr<Type>,
      hashing::Hasher<std::pair<std::vector<StructMember>, std::string>>>
      struct_types_;
  std::mutex struct_mut_;

  // TODO: is_bit_ptr?
  std::unordered_map<std::pair<const Type *, bool>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::pair<const Type *, bool>>>
      pointer_types_;
  std::mutex pointer_mut_;

  std::unordered_map<std::tuple<int, bool, const Type *>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::tuple<int, bool, const Type *>>>
      quant_int_types_;
  std::mutex quant_int_mut_;

  std::unordered_map<
      std::tuple<const Type *, const Type *, float64>,
      std::unique_ptr<Type>,
      hashing::Hasher<std::tuple<const Type *, const Type *, float64>>>
      quant_fixed_types_;
  std::mutex quant_fixed_mut_;

  std::unordered_map<
      std::tuple<const Type *, const Type *, const Type *>,
      std::unique_ptr<Type>,
      hashing::Hasher<std::tuple<const Type *, const Type *, const Type *>>>
      quant_float_types_;
  std::mutex quant_float_mut_;

  // TODO: avoid duplication
  std::vector<std::unique_ptr<BitStructType>> bit_struct_types_;
  std::mutex bit_struct_mut_;

  // TODO: avoid duplication
  std::vector<std::unique_ptr<Type>> quant_array_types_;
  std::mutex quant_array_mut_;
};

DataType promoted_type(DataType a, DataType b);

}  // namespace taichi::lang
