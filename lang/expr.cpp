#include "tlang.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

int Node::counter = 0;
std::map<Node::Type, std::string> Node::node_type_names;

Expr &Expr::operator=(const Expr &o) {
  // TC_ASSERT(allow_store);
  if (!allow_store || !node || node->type != NodeType::pointer) {
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
  TC_ASSERT(node->type == NodeType::addr);
  TC_ASSERT(i->type == NodeType::index || i->data_type == DataType::i32);
  return create(Type::pointer, *this, i);
}

bool Expr::allow_store = true;
}

TC_NAMESPACE_END
