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

  Type *get_primitive_type(PrimitiveTypeID id);

  PrimitiveType *get_primitive_int_type(int bits, bool is_signed = true);

  PrimitiveType *get_primitive_real_type(int bits);

  Type *get_tensor_type(std::vector<int> shape, Type *element);

  const Type *get_struct_type(
      const std::vector<AbstractDictionaryMember> &elements,
      const std::string &layout = "none");

  const Type *get_argpack_type(
      const std::vector<AbstractDictionaryMember> &elements,
      const std::string &layout = "none");

  const Type *get_struct_type_for_argpack_ptr(
      DataType dt,
      const std::string &layout = "none");

  const Type *get_ndarray_struct_type(DataType dt,
                                      int ndim,
                                      bool needs_grad = false);

  const Type *get_rwtexture_struct_type();

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
      const std::vector<int> &member_exponents,
      const std::vector<std::vector<int>> &member_exponent_users);

  Type *get_quant_array_type(PrimitiveType *physical_type,
                             Type *element_type,
                             int num_elements);

  static DataType create_tensor_type(std::vector<int> shape, DataType element);

  constexpr static int SHAPE_POS_IN_NDARRAY = 0;
  constexpr static int DATA_PTR_POS_IN_NDARRAY = 1;
  constexpr static int GRAD_PTR_POS_IN_NDARRAY = 2;
  constexpr static int DATA_PTR_POS_IN_ARGPACK = 0;

 private:
  TypeFactory();

  std::unordered_map<PrimitiveTypeID, std::unique_ptr<Type>> primitive_types_;
  std::mutex primitive_mut_;

  std::unordered_map<std::pair<std::string, Type *>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::pair<std::string, Type *>>>
      tensor_types_;
  std::mutex tensor_mut_;

  std::unordered_map<
      std::pair<std::vector<AbstractDictionaryMember>, std::string>,
      std::unique_ptr<Type>,
      hashing::Hasher<
          std::pair<std::vector<AbstractDictionaryMember>, std::string>>>
      struct_types_;
  std::mutex struct_mut_;
  std::unordered_map<std::vector<AbstractDictionaryMember>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::vector<AbstractDictionaryMember>>>
      argpack_types_;
  std::mutex argpack_mut_;

  // TODO: is_bit_ptr?
  std::unordered_map<std::pair<const Type *, bool>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::pair<const Type *, bool>>>
      pointer_types_;
  std::mutex pointer_mut_;

  std::unordered_map<std::tuple<int, bool, Type *>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::tuple<int, bool, Type *>>>
      quant_int_types_;
  std::mutex quant_int_mut_;

  std::unordered_map<std::tuple<Type *, Type *, float64>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::tuple<Type *, Type *, float64>>>
      quant_fixed_types_;
  std::mutex quant_fixed_mut_;

  std::unordered_map<std::tuple<Type *, Type *, Type *>,
                     std::unique_ptr<Type>,
                     hashing::Hasher<std::tuple<Type *, Type *, Type *>>>
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
