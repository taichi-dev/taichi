#include "taichi/ir/stmt_op_types.h"

#include "taichi/common/logging.h"

namespace taichi {
namespace lang {
std::string unary_op_type_name(UnaryOpType type) {
  switch (type) {
#define PER_UNARY_OP(i) \
  case UnaryOpType::i:  \
    return #i;

#include "taichi/inc/unary_op.inc.h"

#undef PER_UNARY_OP
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string binary_op_type_name(BinaryOpType type) {
  switch (type) {
#define PER_BINARY_OP(x) \
  case BinaryOpType::x:  \
    return #x;

#include "taichi/inc/binary_op.inc.h"

#undef PER_BINARY_OP
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string binary_op_type_symbol(BinaryOpType type) {
  switch (type) {
#define REGISTER_TYPE(i, s) \
  case BinaryOpType::i:     \
    return #s;

    REGISTER_TYPE(mul, *);
    REGISTER_TYPE(add, +);
    REGISTER_TYPE(sub, -);
    REGISTER_TYPE(div, /);
    REGISTER_TYPE(truediv, /);
    REGISTER_TYPE(floordiv, /);
    REGISTER_TYPE(mod, %);
    REGISTER_TYPE(max, max);
    REGISTER_TYPE(min, min);
    REGISTER_TYPE(atan2, atan2);
    REGISTER_TYPE(cmp_lt, <);
    REGISTER_TYPE(cmp_le, <=);
    REGISTER_TYPE(cmp_gt, >);
    REGISTER_TYPE(cmp_ge, >=);
    REGISTER_TYPE(cmp_ne, !=);
    REGISTER_TYPE(cmp_eq, ==);
    REGISTER_TYPE(bit_and, &);
    REGISTER_TYPE(bit_or, |);
    REGISTER_TYPE(logical_and, &&);
    REGISTER_TYPE(logical_or, ||);
    REGISTER_TYPE(bit_xor, ^);
    REGISTER_TYPE(pow, pow);
    REGISTER_TYPE(bit_shl, <<);
    REGISTER_TYPE(bit_sar, >>);
    REGISTER_TYPE(bit_shr, shr);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string ternary_type_name(TernaryOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case TernaryOpType::i: \
    return #i;

    REGISTER_TYPE(select);
    REGISTER_TYPE(ifte);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string atomic_op_type_name(AtomicOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case AtomicOpType::i:  \
    return #i;

    REGISTER_TYPE(add);
    REGISTER_TYPE(sub);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(bit_and);
    REGISTER_TYPE(bit_or);
    REGISTER_TYPE(bit_xor);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

BinaryOpType atomic_to_binary_op_type(AtomicOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case AtomicOpType::i:  \
    return BinaryOpType::i;

    REGISTER_TYPE(add);
    REGISTER_TYPE(sub);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(bit_and);
    REGISTER_TYPE(bit_or);
    REGISTER_TYPE(bit_xor);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string snode_op_type_name(SNodeOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case SNodeOpType::i:   \
    return #i;

    REGISTER_TYPE(is_active);
    REGISTER_TYPE(length);
    REGISTER_TYPE(get_addr);
    REGISTER_TYPE(activate);
    REGISTER_TYPE(deactivate);
    REGISTER_TYPE(append);
    REGISTER_TYPE(clear);
    REGISTER_TYPE(undefined);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

std::string texture_op_type_name(TextureOpType type) {
  switch (type) {
#define REGISTER_TYPE(i) \
  case TextureOpType::i: \
    return #i;

    REGISTER_TYPE(sample_lod);
    REGISTER_TYPE(fetch_texel);
    REGISTER_TYPE(undefined);

#undef REGISTER_TYPE
    default:
      TI_NOT_IMPLEMENTED
  }
}

}  // namespace lang
}  // namespace taichi
