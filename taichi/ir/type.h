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

  std::string to_string() const override;

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

  std::string to_string() const override;

 private:
  int num_elements_{0};
  Type *element_{nullptr};
};

class CustomIntType : public Type {
 public:
  CustomIntType(int num_bits,
                bool is_signed,
                Type *compute_type = nullptr,
                Type *physical_type = nullptr);

  std::string to_string() const override;

  void set_physical_type(Type *physical_type) {
    this->physical_type_ = physical_type;
  }

  Type *get_physical_type() {
    return physical_type_;
  }

  Type *get_compute_type() {
    return compute_type_;
  }

  int get_num_bits() const {
    return num_bits_;
  }

  bool get_is_signed() const {
    return is_signed_;
  }

 private:
  // TODO(type): for now we can uniformly use i32 as the "compute_type". It may
  // be a good idea to make "compute_type" also customizable.
  Type *compute_type_{nullptr};
  Type *physical_type_{nullptr};
  int num_bits_{32};
  bool is_signed_{true};
};

class CustomFloatType : public Type {
 public:
  CustomFloatType(Type *digits_type, Type *compute_type, float64 scale);

  std::string to_string() const override;

  float64 get_scale() const {
    return scale_;
  }

  Type *get_digits_type() {
    return digits_type_;
  }

  Type *get_compute_type() {
    return compute_type_;
  }

 private:
  float64 scale_;
  Type *digits_type_{nullptr};
  Type *compute_type_{nullptr};
};

class BitStructType : public Type {
 public:
  BitStructType(PrimitiveType *physical_type,
                std::vector<Type *> member_types,
                std::vector<int> member_bit_offsets);

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

class BitArrayType : public Type {
 public:
  BitArrayType(PrimitiveType *physical_type,
               Type *element_type_,
               int num_elements_)
      : physical_type_(physical_type),
        element_type_(element_type_),
        num_elements_(num_elements_) {
    // TODO: avoid assertion?
    TI_ASSERT(element_type_->is<CustomIntType>());
    element_num_bits_ = element_type_->as<CustomIntType>()->get_num_bits();
  }

  std::string to_string() const override;

  PrimitiveType *get_physical_type() const {
    return physical_type_;
  }

  Type *get_element_type() const {
    return element_type_;
  }

  int get_num_elements() const {
    return num_elements_;
  }

  int get_element_num_bits() const {
    return element_num_bits_;
  }

 private:
  PrimitiveType *physical_type_;
  Type *element_type_;
  int num_elements_;
  int element_num_bits_;
};

TLANG_NAMESPACE_END
