#include "taichi/ir/type.h"
#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Note: these primitive types should never be freed. They are supposed to live
// together with the process. This is a temporary solution. Later we should
// manage its ownership more systematically.

// This part doesn't look good, but we will remove it soon anyway.
#define PER_TYPE(x)                                            \
  DataType PrimitiveType::x =                                  \
      DataType(Program::get_type_factory().get_primitive_type( \
          PrimitiveType::primitive_type::x));

#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

DataType::DataType() : ptr_(PrimitiveType::unknown.ptr_) {
}

DataType PrimitiveType::get(PrimitiveType::primitive_type t) {
  if (false) {
  }
#define PER_TYPE(x) else if (t == primitive_type::x) return PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  else {
    TI_NOT_IMPLEMENTED
  }
}

std::size_t DataType::hash() const {
  if (auto primitive = dynamic_cast<const PrimitiveType *>(ptr_)) {
    return (std::size_t)primitive->type;
  } else {
    TI_NOT_IMPLEMENTED
  }
}

std::string PrimitiveType::to_string() const {
  return data_type_name(DataType(this));
}

TLANG_NAMESPACE_END
