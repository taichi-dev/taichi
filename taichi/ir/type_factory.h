#pragma once

#include "taichi/lang_util.h"

#include <mutex>

TLANG_NAMESPACE_BEGIN

class TypeFactory {
 public:
  static TypeFactory &get_instance();

  Type *get_primitive_type(PrimitiveTypeID id);

  Type *get_primitive_int_type(int bits, bool is_signed = true);

  Type *get_vector_type(int num_elements, Type *element);

  Type *get_pointer_type(Type *element);

  Type *get_custom_int_type(int num_bits, bool is_signed);

  Type *get_bit_struct(int container_bits,
                       std::vector<Type *> member_types,
                       std::vector<int> member_bit_offsets);

 private:
  TypeFactory();

  std::unordered_map<PrimitiveTypeID, std::unique_ptr<Type>> primitive_types_;

  // TODO: use unordered map
  std::map<std::pair<int, Type *>, std::unique_ptr<Type>> vector_types_;

  // TODO: is_bit_ptr?
  std::map<Type *, std::unique_ptr<Type>> pointer_types_;

  // TODO: use unordered map
  std::map<std::pair<int, bool>, std::unique_ptr<Type>> custom_int_types_;

  // TODO: avoid duplication
  std::vector<std::unique_ptr<Type>> bit_struct_types_;

  std::mutex mut_;
};

TLANG_NAMESPACE_END
