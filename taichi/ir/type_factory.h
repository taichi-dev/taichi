#pragma once

#include "taichi/lang_util.h"

#include <mutex>

TLANG_NAMESPACE_BEGIN

class TypeFactory {
 public:
  Type *get_primitive_type(PrimitiveType::primitive_type id);

  Type *get_vector_type(int num_elements, Type *element);

  Type *get_pointer_type(Type *element);

 private:
  std::unordered_map<PrimitiveType::primitive_type, std::unique_ptr<Type>>
      primitive_types_;

  // TODO: use unordered map
  std::map<std::pair<int, Type *>, std::unique_ptr<Type>> vector_types_;

  // TODO: is_bit_ptr?
  std::map<Type *, std::unique_ptr<Type>> pointer_types_;

  std::mutex mut_;
};

TLANG_NAMESPACE_END
