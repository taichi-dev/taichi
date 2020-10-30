#pragma once

#include "taichi/common/core.h"
#include "taichi/util/bit.h"

TLANG_NAMESPACE_BEGIN

enum class PrimitiveTypeID : int {
#define PER_TYPE(x) x,
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
};

class Type {
 public:
  virtual std::string to_string() const = 0;

  template <typename T>
  bool is() const {
    return cast<T>() != nullptr;
  }

  template <typename T>
  const T *cast() const {
    return dynamic_cast<const T *>(this);
  }

  template <typename T>
  T *cast() {
    return dynamic_cast<T *>(this);
  }

  template <typename T>
  T *as() {
    auto p = dynamic_cast<T *>(this);
    TI_ASSERT(p != nullptr);
    return p;
  }

  int vector_width() const;

  bool is_primitive(PrimitiveTypeID type) const;

  virtual ~Type() {
  }
};

// A "Type" handle. This should be removed later.
class DataType {
 public:
  DataType();

  DataType(Type *ptr) : ptr_(ptr) {
  }

  DataType(const DataType &o) : ptr_(o.ptr_) {
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
  Type *get_ptr() const {
    return ptr_;
  }

  // Temporary API and members
  // for LegacyVectorType-compatibility

  Type *operator->() const {
    return ptr_;
  }

  DataType &operator=(const DataType &o) {
    ptr_ = o.ptr_;
    return *this;
  }

  bool is_pointer() const;

  void set_is_pointer(bool ptr);

  DataType ptr_removed() const;

 private:
  Type *ptr_;
};

// Note that all types are immutable once created.

class PrimitiveType : public Type {
 public:
#define PER_TYPE(x) static DataType x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

  // TODO(type): make 'type' private and add a const getter
  PrimitiveTypeID type;

  PrimitiveType(PrimitiveTypeID type) : type(type) {
  }

  std::string to_string() const override;

  static DataType get(PrimitiveTypeID type);
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
    TI_ASSERT(num_elements_ != 1);
  }

  Type *get_element_type() const {
    return element_;
  }

  int get_num_elements() const {
    return num_elements_;
  }

  std::string to_string() const override {
    return fmt::format("[{} x {}]", num_elements_, element_->to_string());
  }

 private:
  int num_elements_{0};
  Type *element_{nullptr};
};

class CustomIntType : public Type {
 public:
  CustomIntType(int num_bits, bool is_signed)
      : num_bits_(num_bits), is_signed_(is_signed) {
  }

  std::string to_string() const override;

  int get_num_bits() const {
    return num_bits_;
  }

  bool get_is_signed() const {
    return is_signed_;
  }

 private:
  // TODO(type): for now we can uniformly use i32 as the "compute_type". It may
  // be a good idea to make "compute_type" also customizable.
  int num_bits_;
  bool is_signed_;
};

class BitStructType : public Type {
 public:
  BitStructType(PrimitiveType *physical_type,
                std::vector<Type *> member_types,
                std::vector<int> member_bit_offsets)
      : physical_type_(physical_type),
        member_types_(member_types),
        member_bit_offsets_(member_bit_offsets) {
    TI_ASSERT(member_types_.size() == member_bit_offsets_.size());
  }

  std::string to_string() const override;

  PrimitiveType *get_physical_type() const {
    return physical_type_;
  }

  int get_num_memebrs() const {
    return (int)member_types_.size();
  }

  Type *get_member_type(int i) const {
    return member_types_[i];
  }

  int get_member_bit_offset(int i) const {
    return member_bit_offsets_[i];
  }

 private:
  PrimitiveType *physical_type_;
  std::vector<Type *> member_types_;
  std::vector<int> member_bit_offsets_;
};

TLANG_NAMESPACE_END
