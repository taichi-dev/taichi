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

class PointerType : public Type {
 public:
  PointerType(Type *pointee, bool is_bit_pointer)
      : pointee_(pointee), is_bit_pointer_(is_bit_pointer) {
  }

  Type *get_pointee_type() const {
    return pointee_;
  }

  auto get_addr_space() const {
    return addr_space_;
  }

  bool is_bit_pointer() const {
    return is_bit_pointer_;
  }

  std::string to_string() const override {
    return fmt::format("*{}", pointee_->to_string());
  };

 private:
  Type *pointee_{nullptr};
  int addr_space_{0};  // TODO: make this an enum
  bool is_bit_pointer_{false};
};

class VectorType : public Type {
 public:
  VectorType(int num_elements, Type *element)
      : num_elements_(num_elements), element_(element) {
  }

  Type *get_element_type() const {
    return element_;
  }

  int get_num_elements() const {
    return num_elements_;
  }

  std::string to_string() const override {
    return fmt::format("[{} x {}]", num_elements_, element_->to_string());
  };

 private:
  int num_elements_{0};
  Type *element_{nullptr};
};

TLANG_NAMESPACE_END
