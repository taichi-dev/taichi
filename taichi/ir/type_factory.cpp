#include "taichi/ir/type_factory.h"

TLANG_NAMESPACE_BEGIN

Type *TypeFactory::get_primitive_type(
    taichi::lang::PrimitiveType::primitive_type id) {
  std::lock_guard<std::mutex> _(mut_);

  if (primitive_types_.find(id) == primitive_types_.end()) {
    primitive_types_[id] = std::make_unique<PrimitiveType>(id);
  }

  return primitive_types_[id].get();
}

TLANG_NAMESPACE_END
