#include "tlang.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

std::map<Node::DataType, std::string> Node::data_type_names;
std::map<Node::Type, std::string> Node::node_type_names;

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

Expr Expr::operator[](const Expr &i) {
  return create(Type::pointer, *this, i);
}

}

TC_NAMESPACE_END
