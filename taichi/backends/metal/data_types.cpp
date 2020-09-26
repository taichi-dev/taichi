#include "taichi/backends/metal/data_types.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

MetalDataType to_metal_type(DataType dt) {
  switch (dt) {
#define METAL_CASE(x) \
  case DataType::x:   \
    return MetalDataType::x

    METAL_CASE(f32);
    METAL_CASE(f64);
    METAL_CASE(i8);
    METAL_CASE(i16);
    METAL_CASE(i32);
    METAL_CASE(i64);
    METAL_CASE(u8);
    METAL_CASE(u16);
    METAL_CASE(u32);
    METAL_CASE(u64);
    METAL_CASE(unknown);
#undef METAL_CASE

    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return MetalDataType::unknown;
}

std::string metal_data_type_name(MetalDataType dt) {
  switch (dt) {
    case MetalDataType::f32:
      return "float";
    case MetalDataType::f64:
      return "double";
    case MetalDataType::i8:
      return "int8_t";
    case MetalDataType::i16:
      return "int16_t";
    case MetalDataType::i32:
      return "int32_t";
    case MetalDataType::i64:
      return "int64_t";
    case MetalDataType::u8:
      return "uint8_t";
    case MetalDataType::u16:
      return "uint16_t";
    case MetalDataType::u32:
      return "uint32_t";
    case MetalDataType::u64:
      return "uint64_t";
    case MetalDataType::unknown:
      return "unknown";
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return "";
}

size_t metal_data_type_bytes(MetalDataType dt) {
  switch (dt) {
    case MetalDataType::f32:
      return 4;
    case MetalDataType::f64:
      return 8;
    case MetalDataType::i8:
      return 1;
    case MetalDataType::i16:
      return 2;
    case MetalDataType::i32:
      return 4;
    case MetalDataType::i64:
      return 8;
    case MetalDataType::u8:
      return 1;
    case MetalDataType::u16:
      return 2;
    case MetalDataType::u32:
      return 4;
    case MetalDataType::u64:
      return 8;
    default:
      TI_NOT_IMPLEMENTED;
      break;
  }
  return 0;
}

std::string metal_unary_op_type_symbol(UnaryOpType type) {
  switch (type) {
    case UnaryOpType::neg:
      return "-";
    case UnaryOpType::sqrt:
      return "sqrt";
    case UnaryOpType::floor:
      return "floor";
    case UnaryOpType::ceil:
      return "ceil";
    case UnaryOpType::abs:
      return "abs";
    case UnaryOpType::sgn:
      return "sign";
    case UnaryOpType::sin:
      return "sin";
    case UnaryOpType::asin:
      return "asin";
    case UnaryOpType::cos:
      return "cos";
    case UnaryOpType::acos:
      return "acos";
    case UnaryOpType::tan:
      return "tan";
    case UnaryOpType::tanh:
      return "tanh";
    case UnaryOpType::exp:
      return "exp";
    case UnaryOpType::log:
      return "log";
    case UnaryOpType::rsqrt:
      return "rsqrt";
    case UnaryOpType::bit_not:
      return "~";
    case UnaryOpType::logic_not:
      return "!";
    // case UnaryOpType::inv:
    // case UnaryOpType::rcp:
    // case UnaryOpType::undefined:
    default:
      TI_NOT_IMPLEMENTED;
  }
  return "";
}

}  // namespace metal
TLANG_NAMESPACE_END
