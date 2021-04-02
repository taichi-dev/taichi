#include "taichi/ir/snode_types.h"

#include "taichi/common/logging.h"

namespace taichi {
namespace lang {

std::string snode_type_name(SNodeType t) {
  switch (t) {
#define PER_SNODE(i) \
  case SNodeType::i: \
    return #i;

#include "taichi/inc/snodes.inc.h"

#undef PER_SNODE
    default:
      TI_NOT_IMPLEMENTED;
  }
}

bool is_gc_able(SNodeType t) {
  return (t == SNodeType::pointer || t == SNodeType::dynamic);
}

}  // namespace lang
}  // namespace taichi
