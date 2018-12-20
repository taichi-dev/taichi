#include "tlang.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

int Node::counter = 0;
std::map<Node::Type, std::string> Node::node_type_names;

Expr &Expr::operator=(const Expr &o) {
  if (!node || node->type != NodeType::pointer) {
    // Expr assignment
    node = o.node;
  } else {
    // store to pointed addr
    TC_ASSERT(node->type == NodeType::pointer);
    auto &prog = get_current_program();
    // TC_ASSERT(&prog != nullptr);
    // TC_ASSERT(node->get_address().initialized());
    prog.store(*this, load_if_pointer(o));
  }
  return *this;
}

Expr Expr::operator[](const Expr &i) {
  TC_ASSERT(i);
  TC_ASSERT(i->type == NodeType::index || i->data_type == DataType::i32);
  return create(Type::pointer, *this, i);
}

}

TC_NAMESPACE_END
