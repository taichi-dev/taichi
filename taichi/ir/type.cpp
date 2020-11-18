#include "taichi/ir/type.h"

#include "taichi/program/program.h"

TLANG_NAMESPACE_BEGIN

// Note: these primitive types should never be freed. They are supposed to live
// together with the process. This is a temporary solution. Later we should
// manage its ownership more systematically.

// This part doesn't look good, but we will remove it soon anyway.
#define PER_TYPE(x)                     \
  DataType PrimitiveType::x = DataType( \
      TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::x));

#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE

DataType::DataType() : ptr_(PrimitiveType::unknown.ptr_) {
}

DataType PrimitiveType::get(PrimitiveTypeID t) {
  if (false) {
  }
#define PER_TYPE(x) else if (t == PrimitiveTypeID::x) return PrimitiveType::x;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  else {
    TI_NOT_IMPLEMENTED
  }
}

std::size_t DataType::hash() const {
  if (auto primitive = ptr_->cast<PrimitiveType>()) {
    return (std::size_t)primitive->type;
  } else if (auto pointer = ptr_->cast<PointerType>()) {
    return 10007 + DataType(pointer->get_pointee_type()).hash();
  } else {
    TI_NOT_IMPLEMENTED
  }
}

bool DataType::is_pointer() const {
  return ptr_->is<PointerType>();
}

void DataType::set_is_pointer(bool is_ptr) {
  if (is_ptr && !ptr_->is<PointerType>()) {
    ptr_ = TypeFactory::get_instance().get_pointer_type(ptr_);
  }
  if (!is_ptr && ptr_->is<PointerType>()) {
    ptr_ = ptr_->cast<PointerType>()->get_pointee_type();
  }
}

DataType DataType::ptr_removed() const {
  auto t = ptr_;
  auto ptr_type = t->cast<PointerType>();
  if (ptr_type) {
    return DataType(ptr_type->get_pointee_type());
  } else {
    return *this;
  }
}

std::string PrimitiveType::to_string() const {
  return data_type_name(DataType(const_cast<PrimitiveType *>(this)));
}

std::string PointerType::to_string() const {
  if (is_bit_pointer_) {
    // "^" for bit-level pointers
    return fmt::format("^{}", pointee_->to_string());
  } else {
    // "*" for C-style byte-level pointers
    return fmt::format("*{}", pointee_->to_string());
  }
}

std::string VectorType::to_string() const {
  return fmt::format("[{} x {}]", num_elements_, element_->to_string());
}

int Type::vector_width() const {
  if (auto vec = cast<VectorType>()) {
    return vec->get_num_elements();
  } else {
    return 1;
  }
}

bool Type::is_primitive(PrimitiveTypeID type) const {
  if (auto p = cast<PrimitiveType>()) {
    return p->type == type;
  } else {
    return false;
  }
}

std::string CustomIntType::to_string() const {
  return fmt::format("c{}{}", is_signed_ ? 'i' : 'u', num_bits_);
}

CustomIntType::CustomIntType(int num_bits, bool is_signed)
    : compute_type(nullptr), physical_type(nullptr),
      num_bits_(num_bits), is_signed_(is_signed) {
  // TODO(type): support customizable compute_type
  //  and should we expose it to users?
  TI_ASSERT(num_bits <= 32);
  if (is_signed) {
    compute_type =
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::i32);
  } else {
    compute_type =
        TypeFactory::get_instance().get_primitive_type(PrimitiveTypeID::u32);
  }
}

#define SET_COMPUTE_TYPE(n, N)         \
  else if (n == N) {                   \
    if (is_signed)                     \
      type_id = PrimitiveTypeID::i##N; \
    else                               \
      type_id = PrimitiveTypeID::u##N; \
  }

CustomIntType::CustomIntType(int compute_type_bits,
                             int num_bits,
                             bool is_signed)
    : compute_type(nullptr), physical_type(nullptr),
      num_bits_(num_bits), is_signed_(is_signed) {
  auto type_id = PrimitiveTypeID::unknown;
  if (false) {
  }
  SET_COMPUTE_TYPE(compute_type_bits, 64)
  SET_COMPUTE_TYPE(compute_type_bits, 32)
  SET_COMPUTE_TYPE(compute_type_bits, 16)
  SET_COMPUTE_TYPE(compute_type_bits, 8)
  else {TI_NOT_IMPLEMENTED} compute_type =
      TypeFactory::get_instance().get_primitive_type(type_id);
}

CustomIntType::CustomIntType(int compute_type_bits,
                             Type *physical_type,
                             int num_bits,
                             bool is_signed)
    : compute_type(nullptr), physical_type(physical_type),
      num_bits_(num_bits), is_signed_(is_signed) {
  auto type_id = PrimitiveTypeID::unknown;
  if (false) {
  }
  SET_COMPUTE_TYPE(compute_type_bits, 64)
  SET_COMPUTE_TYPE(compute_type_bits, 32)
  SET_COMPUTE_TYPE(compute_type_bits, 16)
  SET_COMPUTE_TYPE(compute_type_bits, 8)
  else {TI_NOT_IMPLEMENTED} compute_type =
                                TypeFactory::get_instance().get_primitive_type(type_id);
}

BitStructType::BitStructType(PrimitiveType *physical_type,
                             std::vector<Type *> member_types,
                             std::vector<int> member_bit_offsets)
    : physical_type_(physical_type),
      member_types_(member_types),
      member_bit_offsets_(member_bit_offsets) {
  TI_ASSERT(member_types_.size() == member_bit_offsets_.size());
  int physical_type_bits = data_type_bits(physical_type);
  for (auto i = 0; i < member_types_.size(); ++i) {
    auto bits_end = member_types_[i]->as<CustomIntType>()->get_num_bits() +
                    member_bit_offsets_[i];
    TI_ASSERT(physical_type_bits >= bits_end)
  }
}

std::string BitStructType::to_string() const {
  std::string str = "bs(";
  int num_members = (int)member_bit_offsets_.size();
  for (int i = 0; i < num_members; i++) {
    str += fmt::format("{}@{}", member_types_[i]->to_string(),
                       member_bit_offsets_[i]);
    if (i + 1 < num_members) {
      str += ", ";
    }
  }
  return str + ")";
}

std::string BitArrayType::to_string() const {
  return fmt::format("ba({}x{})", element_type_->to_string(), num_elements_);
}

TLANG_NAMESPACE_END
