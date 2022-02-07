#include "taichi/backends/opengl/opengl_utils.h"
#include "glad/gl.h"

namespace taichi {
namespace lang {
namespace opengl {

uint32_t to_gl_dtype_enum(DataType dt) {
  if (dt == PrimitiveType::u64) {
    return GL_UNSIGNED_INT64_ARB;
  } else if (dt == PrimitiveType::i64) {
    return GL_INT64_ARB;
  } else if (dt == PrimitiveType::u32) {
    return GL_UNSIGNED_INT;
  } else if (dt == PrimitiveType::i32) {
    return GL_INT;
  } else if (dt == PrimitiveType::u16) {
    return GL_UNSIGNED_SHORT;
  } else if (dt == PrimitiveType::i16) {
    return GL_SHORT;
  } else if (dt == PrimitiveType::u8) {
    return GL_UNSIGNED_BYTE;
  } else if (dt == PrimitiveType::i8) {
    return GL_BYTE;
  } else if (dt == PrimitiveType::f64) {
    return GL_DOUBLE;
  } else if (dt == PrimitiveType::f32) {
    return GL_FLOAT;
  } else {
    TI_NOT_IMPLEMENTED
  }
}
}  // namespace opengl
}  // namespace lang
}  // namespace taichi
