#include "tlang.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

Expr &Expr::operator=(const Expr &o) {
  if (!node || node->type != NodeType::pointer) {
    TC_TAG;
    // Expr assignment
    node = o.node;
    TC_P(node.get());
  } else {
    TC_TAG;
    // store to pointed addr
    auto &prog = get_current_program();
    TC_ASSERT(&prog != nullptr);
    // TC_ASSERT(node->get_address().initialized());
    prog.store(*this, o);
  }
  return *this;
}

Expr &&Expr::operator[](const Expr &i) {
  return create(Type::pointer, *this, i);
}

}

TC_NAMESPACE_END
