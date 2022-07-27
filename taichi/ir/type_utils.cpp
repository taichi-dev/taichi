#include "taichi/ir/type_utils.h"

namespace taichi {
namespace lang {

std::string data_type_name(DataType t) {
  if (!t->is<PrimitiveType>()) {
    return t->to_string();
  }

  // Handle primitive types below.

  if (false) {
  }
#define PER_TYPE(i) else if (t->is_primitive(PrimitiveTypeID::i)) return #i;
#include "taichi/inc/data_type.inc.h"
#undef PER_TYPE
  else
    TI_NOT_IMPLEMENTED
}

int data_type_size(DataType t) {
  // TODO:
  //  1. Ensure in the old code, pointer attributes of t are correct (by
  //  setting a loud failure on pointers);
  //  2. Support pointer types here.
  t.set_is_pointer(false);
  if (false) {
  } else if (t->is_primitive(PrimitiveTypeID::f16))
    return 2;
  else if (t->is_primitive(PrimitiveTypeID::gen))
    return 0;
  else if (t->is_primitive(PrimitiveTypeID::unknown))
    return -1;

#define REGISTER_DATA_TYPE(i, j) \
  else if (t->is_primitive(PrimitiveTypeID::i)) return sizeof(j)

  REGISTER_DATA_TYPE(f32, float32);
  REGISTER_DATA_TYPE(f64, float64);
  REGISTER_DATA_TYPE(i8, int8);
  REGISTER_DATA_TYPE(i16, int16);
  REGISTER_DATA_TYPE(i32, int32);
  REGISTER_DATA_TYPE(i64, int64);
  REGISTER_DATA_TYPE(u8, uint8);
  REGISTER_DATA_TYPE(u16, uint16);
  REGISTER_DATA_TYPE(u32, uint32);
  REGISTER_DATA_TYPE(u64, uint64);

#undef REGISTER_DATA_TYPE
  else {
    TI_NOT_IMPLEMENTED
  }
}

std::string data_type_format(DataType dt) {
  if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return "%hd";
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return "%hu";
  } else if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return "%d";
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return "%u";
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    // Use %lld on Windows.
    // Discussion: https://github.com/taichi-dev/taichi/issues/2522
    return "%lld";
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return "%llu";
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return "%f";
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return "%.12f";
  } else if (dt->is<QuantIntType>()) {
    return "%d";
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    // f16 (and f32) is converted to f64 before printing, see
    // TaskCodeGenLLVM::visit(PrintStmt *stmt) and
    // TaskCodeGenCUDA::visit(PrintStmt *stmt) for more details.
    return "%f";
  } else {
    TI_NOT_IMPLEMENTED
  }
}

}  // namespace lang
}  // namespace taichi
