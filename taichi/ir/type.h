#pragma once

#include "taichi/common/core.h"
#include "taichi/util/bit.h"
#include "taichi/util/hash.h"

namespace taichi::lang {

class TensorType;

enum class PrimitiveTypeID : int {
#define PER_TYPE(x) x,
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
};

enum class TypeKind : int {
#define PER_TYPE_KIND(x) x,
#include "taichi/inc/type_kind.inc.h"
#undef PER_TYPE_KIND
};

class TI_DLL_EXPORT Type {
 public:
  explicit Type(TypeKind type_kind) : type_kind(type_kind) {
  }
  TypeKind type_kind;
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
    TI_ASSERT_INFO(p != nullptr, "Cannot treat {} as {}", this->to_string(),
                   typeid(T).name());
    return p;
  }

  template <typename T>
  const T *as() const {
    auto p = dynamic_cast<const T *>(this);
    TI_ASSERT_INFO(p != nullptr, "Cannot treat {} as {}", this->to_string(),
                   typeid(T).name());
    return p;
  }

  template <typename S, typename T>
  static typename std::enable_if<std::is_base_of_v<Type, T>, void>::type
  ptr_io(const T *&ptr, S &serializer, bool writing);

  using JsonValue = ::liong::json::JsonValue;
  using JsonObject = ::liong::json::JsonObject;

  template <typename T>
  static typename std::enable_if<std::is_base_of_v<Type, T>, void>::type
  jsonserde_ptr_io(const T *&ptr, JsonValue &value, bool writing, bool strict);

  // For serialization
  virtual const Type *get_type() const = 0;

  bool is_primitive(PrimitiveTypeID type) const;

  virtual Type *get_compute_type() {
    TI_NOT_IMPLEMENTED;
  }

  virtual ~Type() {
  }
};

// A "Type" handle. This should be removed later.
class TI_DLL_EXPORT DataType {
 public:
  DataType();

  // NOLINTNEXTLINE(google-explicit-constructor)
  DataType(const Type *ptr) : ptr_((Type *)ptr) {
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

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator const Type *() const {
    return ptr_;
  }

  // NOLINTNEXTLINE(google-explicit-constructor)
  operator Type *() {
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

  std::vector<int> get_shape() const;

  DataType get_element_type() const;

  TI_IO_DEF(ptr_);

 private:
  Type *ptr_;
};

// Note that all types are immutable once created.

class TI_DLL_EXPORT PrimitiveType : public Type {
 public:
#define PER_TYPE(x) static DataType x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

  // TODO(type): make 'type' private and add a const getter
  PrimitiveTypeID type;

  explicit PrimitiveType(PrimitiveTypeID type = PrimitiveTypeID::unknown)
      : Type(TypeKind::Primitive), type(type) {
  }

  std::string to_string() const override;

  Type *get_compute_type() override {
    return this;
  }

  static DataType get(PrimitiveTypeID type);

  const Type *get_type() const override;

  TI_IO_DEF(type);
};

class TI_DLL_EXPORT PointerType : public Type {
 public:
  PointerType() : Type(TypeKind::Pointer){};

  PointerType(Type *pointee, bool is_bit_pointer)
      : Type(TypeKind::Pointer),
        pointee_(pointee),
        is_bit_pointer_(is_bit_pointer) {
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

  const Type *get_type() const override;

  TI_IO_DEF(pointee_, addr_space_, is_bit_pointer_);

 private:
  Type *pointee_{nullptr};
  int addr_space_{0};  // TODO: make this an enum
  bool is_bit_pointer_{false};
};

class TI_DLL_EXPORT TensorType : public Type {
 public:
  TensorType() : Type(TypeKind::Tensor){};
  TensorType(std::vector<int> shape, Type *element)
      : Type(TypeKind::Tensor), shape_(std::move(shape)), element_(element) {
  }

  Type *get_element_type() const {
    return element_;
  }

  int get_num_elements() const {
    int num_elements = 1;
    for (int i = 0; i < (int)shape_.size(); ++i)
      num_elements *= shape_[i];
    return num_elements;
  }

  std::vector<int> get_shape() const {
    return shape_;
  }

  void set_shape(const std::vector<int> &shape) {
    shape_ = shape;
  }

  Type *get_compute_type() override {
    return this;
  }

  std::string to_string() const override;

  size_t get_element_offset(int ind) const;

  const Type *get_type() const override;

  TI_IO_DEF(shape_, element_);

 private:
  std::vector<int> shape_;
  Type *element_{nullptr};
};

struct TI_DLL_EXPORT AbstractDictionaryMember {
  const Type *type;
  std::string name;
  size_t offset{0};
  bool operator==(const AbstractDictionaryMember &other) const {
    return type == other.type && name == other.name && offset == other.offset;
  }
  TI_IO_DEF(type, name, offset)
};

class TI_DLL_EXPORT AbstractDictionaryType : public Type {
 public:
  explicit AbstractDictionaryType(TypeKind type_kind) : Type(type_kind){};
  explicit AbstractDictionaryType(
      TypeKind type_kind,
      const std::vector<AbstractDictionaryMember> &elements,
      const std::string &layout = "none")
      : Type(type_kind), elements_(elements), layout_(layout) {
  }

  const std::string &get_layout() const {
    return layout_;
  }

  const std::vector<AbstractDictionaryMember> &elements() const {
    return elements_;
  }

  Type *get_compute_type() override {
    return this;
  }

  const Type *get_element_type(const std::vector<int> &indices) const;

  TI_IO_DEF(elements_, layout_);

 protected:
  std::vector<AbstractDictionaryMember> elements_;
  std::string layout_;
};

class TI_DLL_EXPORT StructType : public AbstractDictionaryType {
 public:
  StructType() : AbstractDictionaryType(TypeKind::Struct){};
  explicit StructType(const std::vector<AbstractDictionaryMember> &elements,
                      const std::string &layout = "none")
      : AbstractDictionaryType(TypeKind::Struct, elements, layout) {
  }

  std::string to_string() const override;

  size_t get_element_offset(const std::vector<int> &indices) const;

  int get_flattened_num_elements() const {
    int num = 0;
    for (const auto &element : elements_) {
      if (auto struct_type = element.type->cast<StructType>()) {
        num += struct_type->get_flattened_num_elements();
      } else if (auto tensor_type = element.type->cast<TensorType>()) {
        num += tensor_type->get_num_elements();
      } else {
        TI_ASSERT(element.type->is<PrimitiveType>() ||
                  element.type->is<PointerType>());
        num += 1;
      }
    }
    return num;
  }

  const Type *get_type() const override;

  TI_IO_DEF(elements_, layout_);
};

class TI_DLL_EXPORT ArgPackType : public AbstractDictionaryType {
 public:
  ArgPackType() : AbstractDictionaryType(TypeKind::ArgPack){};
  explicit ArgPackType(const std::vector<AbstractDictionaryMember> &elements,
                       const std::string &layout = "none")
      : AbstractDictionaryType(TypeKind::ArgPack, elements, layout) {
  }

  size_t get_element_offset(const std::vector<int> &indices) const;

  std::string to_string() const override;

  const Type *get_type() const override;

  TI_IO_DEF(elements_, layout_);
};

class TI_DLL_EXPORT QuantIntType : public Type {
 public:
  QuantIntType() : Type(TypeKind::QuantInt){};
  QuantIntType(int num_bits, bool is_signed, Type *compute_type = nullptr);

  std::string to_string() const override;

  Type *get_compute_type() override {
    return compute_type_;
  }

  int get_num_bits() const {
    return num_bits_;
  }

  bool get_is_signed() const {
    return is_signed_;
  }

  const Type *get_type() const override;

  TI_IO_DEF(num_bits_, is_signed_, compute_type_);

 private:
  // TODO(type): for now we can uniformly use i32 as the "compute_type". It may
  // be a good idea to make "compute_type" also customizable.
  Type *compute_type_{nullptr};
  int num_bits_{32};
  bool is_signed_{true};
};

class TI_DLL_EXPORT QuantFixedType : public Type {
 public:
  QuantFixedType() : Type(TypeKind::QuantFixed){};
  QuantFixedType(Type *digits_type, Type *compute_type, float64 scale);

  std::string to_string() const override;

  bool get_is_signed() const;

  Type *get_digits_type() {
    return digits_type_;
  }

  Type *get_compute_type() override {
    return compute_type_;
  }

  float64 get_scale() const {
    return scale_;
  }

  const Type *get_type() const override;

  TI_IO_DEF(digits_type_, compute_type_, scale_);

 private:
  Type *digits_type_{nullptr};
  Type *compute_type_{nullptr};
  float64 scale_{1.0};
};

class TI_DLL_EXPORT QuantFloatType : public Type {
 public:
  QuantFloatType() : Type(TypeKind::QuantFloat) {
  }
  QuantFloatType(Type *digits_type, Type *exponent_type, Type *compute_type);

  std::string to_string() const override;

  Type *get_digits_type() {
    return digits_type_;
  }

  Type *get_exponent_type() {
    return exponent_type_;
  }

  int get_exponent_conversion_offset() const;

  int get_digit_bits() const;

  bool get_is_signed() const;

  Type *get_compute_type() override {
    return compute_type_;
  }

  const Type *get_type() const override;

  TI_IO_DEF(digits_type_, exponent_type_, compute_type_);

 private:
  Type *digits_type_{nullptr};
  Type *exponent_type_{nullptr};
  Type *compute_type_{nullptr};
};

class TI_DLL_EXPORT BitStructType : public Type {
 public:
  BitStructType() : Type(TypeKind::BitStruct){};
  BitStructType(PrimitiveType *physical_type,
                const std::vector<Type *> &member_types,
                const std::vector<int> &member_bit_offsets,
                const std::vector<int> &member_exponents,
                const std::vector<std::vector<int>> &member_exponent_users);

  std::string to_string() const override;

  PrimitiveType *get_physical_type() const {
    return physical_type_;
  }

  int get_num_members() const {
    return (int)member_types_.size();
  }

  Type *get_member_type(int i) const {
    return member_types_[i];
  }

  int get_member_bit_offset(int i) const {
    return member_bit_offsets_[i];
  }

  bool get_member_owns_shared_exponent(int i) const {
    return member_exponents_[i] != -1 &&
           member_exponent_users_[member_exponents_[i]].size() > 1;
  }

  int get_member_exponent(int i) const {
    return member_exponents_[i];
  }

  const std::vector<int> &get_member_exponent_users(int i) const {
    return member_exponent_users_[i];
  }

  const Type *get_type() const override;

  TI_IO_DEF(physical_type_,
            member_types_,
            member_bit_offsets_,
            member_exponents_,
            member_exponent_users_);

 private:
  PrimitiveType *physical_type_;
  std::vector<Type *> member_types_;
  std::vector<int> member_bit_offsets_;
  std::vector<int> member_exponents_;
  std::vector<std::vector<int>> member_exponent_users_;
};

class TI_DLL_EXPORT QuantArrayType : public Type {
 public:
  QuantArrayType() : Type(TypeKind::QuantArray) {
  }
  QuantArrayType(PrimitiveType *physical_type,
                 Type *element_type_,
                 int num_elements_)
      : Type(TypeKind::QuantArray),
        physical_type_(physical_type),
        element_type_(element_type_),
        num_elements_(num_elements_) {
    if (auto qit = element_type_->cast<QuantIntType>()) {
      element_num_bits_ = qit->get_num_bits();
    } else if (auto qfxt = element_type_->cast<QuantFixedType>()) {
      element_num_bits_ =
          qfxt->get_digits_type()->as<QuantIntType>()->get_num_bits();
    } else {
      TI_ERROR("Quant array only supports quant int/fixed type for now.");
    }
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

  const Type *get_type() const override;

  TI_IO_DEF(physical_type_, element_type_, num_elements_, element_num_bits_);

 private:
  PrimitiveType *physical_type_;
  Type *element_type_;
  int num_elements_;
  int element_num_bits_;
};

class TypedConstant {
 public:
  DataType dt;

  /*
    Note: Float16 is represented by "float32 value" + "dt = PrimitiveType::f16".
  */
  union {
    uint64 value_bits;
    int32 val_i32;
    float32 val_f32;
    int64 val_i64;
    float64 val_f64;
    int8 val_i8;
    int16 val_i16;
    uint1 val_u1;
    uint8 val_u8;
    uint16 val_u16;
    uint32 val_u32;
    uint64 val_u64;
  };

 public:
  TypedConstant() : dt(PrimitiveType::unknown) {
  }

  explicit TypedConstant(DataType dt) : dt(dt) {
    TI_ASSERT_INFO(dt->is<PrimitiveType>(),
                   "TypedConstant can only be PrimitiveType, got {}",
                   dt->to_string());
    value_bits = 0;
  }

  explicit TypedConstant(int32 x) : dt(PrimitiveType::i32), val_i32(x) {
  }

  explicit TypedConstant(float32 x) : dt(PrimitiveType::f32), val_f32(x) {
  }

  explicit TypedConstant(int64 x) : dt(PrimitiveType::i64), val_i64(x) {
  }

  explicit TypedConstant(float64 x) : dt(PrimitiveType::f64), val_f64(x) {
  }

  explicit TypedConstant(int8 x) : dt(PrimitiveType::i8), val_i8(x) {
  }

  explicit TypedConstant(int16 x) : dt(PrimitiveType::i16), val_i16(x) {
  }

  explicit TypedConstant(uint1 x) : dt(PrimitiveType::u1), val_u1(x) {
  }

  explicit TypedConstant(uint8 x) : dt(PrimitiveType::u8), val_u8(x) {
  }

  explicit TypedConstant(uint16 x) : dt(PrimitiveType::u16), val_u16(x) {
  }

  explicit TypedConstant(uint32 x) : dt(PrimitiveType::u32), val_u32(x) {
  }

  explicit TypedConstant(uint64 x) : dt(PrimitiveType::u64), val_u64(x) {
  }

  template <typename T>
  TypedConstant(DataType dt, const T &value) : dt(dt) {
    // TODO: loud failure on pointers
    dt.set_is_pointer(false);
    if (dt->is_primitive(PrimitiveTypeID::f32)) {
      val_f32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
      val_i32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
      val_i64 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
      val_f64 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
      val_f32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
      val_i8 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
      val_i16 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
      val_u1 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
      val_u8 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
      val_u16 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
      val_u32 = value;
    } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
      val_u64 = value;
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  template <typename T>
  bool equal_value(const T &value) const {
    return equal_type_and_value(TypedConstant(dt, value));
  }

  std::string stringify() const;

  bool equal_type_and_value(const TypedConstant &o) const;

  bool operator==(const TypedConstant &o) const {
    return equal_type_and_value(o);
  }

  int32 &val_int32();
  float32 &val_float32();
  float32 &val_float16();
  int64 &val_int64();
  float64 &val_float64();
  int8 &val_int8();
  int16 &val_int16();
  uint1 &val_uint1();
  uint8 &val_uint8();
  uint16 &val_uint16();
  uint32 &val_uint32();
  uint64 &val_uint64();
  int64 val_int() const;
  uint64 val_uint() const;
  float64 val_float() const;
  int64 val_as_int64() const;  // unifies val_int() and val_uint()
  float64 val_cast_to_float64() const;
};

template <typename S, typename T>
typename std::enable_if<std::is_base_of_v<Type, T>, void>::type
Type::ptr_io(const T *&ptr, S &serializer, bool writing) {
  if (writing) {
    if (ptr == nullptr) {
      serializer("type_kind", (TypeKind)-1);
      return;
    }
    serializer("type_kind", ptr->type_kind);
    switch (ptr->type_kind) {
#define PER_TYPE_KIND(x)                                 \
  case TypeKind::x: {                                    \
    serializer("content", *ptr->template as<x##Type>()); \
    break;                                               \
  }
#include "taichi/inc/type_kind.inc.h"
#undef PER_TYPE_KIND
      default:
        TI_NOT_IMPLEMENTED;
    }
  } else {
    TypeKind type_kind = (TypeKind)-1;
    serializer("type_kind", type_kind);
    if (type_kind == (TypeKind)-1) {
      ptr = nullptr;
      return;
    }
    switch (type_kind) {
#define PER_TYPE_KIND(x)               \
  case TypeKind::x: {                  \
    x##Type content;                   \
    serializer("content", content);    \
    ptr = content.get_type()->as<T>(); \
    break;                             \
  }
#include "taichi/inc/type_kind.inc.h"
#undef PER_TYPE_KIND
      default:
        TI_NOT_IMPLEMENTED;
    }
  }
}

template <typename T>
typename std::enable_if<std::is_base_of_v<Type, T>, void>::type
Type::jsonserde_ptr_io(const T *&ptr,
                       JsonValue &value,
                       bool writing,
                       bool strict) {
  if (writing) {
    if (ptr == nullptr) {
      value = JsonValue(nullptr);
      return;
    }
    JsonObject obj;
    obj.inner["type_kind"] = JsonValue((int)ptr->type_kind);
    JsonValue content;

    switch (ptr->type_kind) {
#define PER_TYPE_KIND(x)                                            \
  case TypeKind::x: {                                               \
    content = ptr->template as<x##Type>()->json_serialize_fields(); \
    break;                                                          \
  }
#include "taichi/inc/type_kind.inc.h"
#undef PER_TYPE_KIND
      default:
        TI_NOT_IMPLEMENTED
    }

    obj.inner["content"] = std::move(content);
    value = JsonValue(std::move(obj));
  } else {
    if (value.is_null()) {
      ptr = nullptr;
      return;
    }
    TypeKind type_kind = (TypeKind)(int)value["type_kind"];
    switch (type_kind) {
#define PER_TYPE_KIND(x)                                      \
  case TypeKind::x: {                                         \
    x##Type content;                                          \
    auto &content_val = value["content"];                     \
    TI_ASSERT(content_val.is_obj());                          \
    content.json_deserialize_fields(content_val.obj, strict); \
    ptr = content.get_type()->as<T>();                        \
    break;                                                    \
  }
#include "taichi/inc/type_kind.inc.h"
#undef PER_TYPE_KIND
      default:
        TI_NOT_IMPLEMENTED
    }
  }
}
}  // namespace taichi::lang

namespace taichi::hashing {

template <>
struct Hasher<lang::AbstractDictionaryMember> {
 public:
  size_t operator()(lang::AbstractDictionaryMember const &member) const {
    size_t ret = hash_value(member.type);
    hash_combine(ret, member.name);
    hash_combine(ret, member.offset);
    return ret;
  }
};

}  // namespace taichi::hashing
