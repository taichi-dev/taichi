#pragma once

#include "taichi/common/core.h"

TLANG_NAMESPACE_BEGIN

class Type {
 public:
  virtual std::string to_string() const = 0;
  virtual ~Type() {
  }
};

// A "Type" handle. This should be removed later.
class DataType {
 public:
  DataType();

  DataType(const Type *ptr) : ptr_(ptr) {
  }

  bool operator==(const DataType &o) const {
    return ptr_ == o.ptr_;
  }

  bool operator!=(const DataType &o) const {
    return !(*this == o);
  }

  std::size_t hash() const;

  std::string to_string() const {
    return ptr_->to_string();
  };

  // TODO: DataType itself should be a pointer in the future
  const Type *get_ptr() const {
    return ptr_;
  }

 private:
  const Type *ptr_;
};

class PrimitiveType : public Type {
 public:
  enum class primitive_type : int {
#define PER_TYPE(x) x,
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  };

#define PER_TYPE(x) static DataType x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

  primitive_type type;

  PrimitiveType(primitive_type type) : type(type) {
  }

  std::string to_string() const override;

  static DataType get(primitive_type type);
};

TLANG_NAMESPACE_END
