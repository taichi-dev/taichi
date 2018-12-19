#include "tlang.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

Expr &Expr::operator=(const Expr &o) {
  if (!node || node->type != NodeType::addr) {
    // Expr assignment
    node = o.node;
  } else {
    // store
    auto &prog = get_current_program();
    TC_ASSERT(&prog != nullptr);
    // TC_ASSERT(node->get_address().initialized());
    prog.store(*this, o);
  }
  return *this;
}

Expr Expr::operator[](Index i) {
}
}
TC_NAMESPACE_END
