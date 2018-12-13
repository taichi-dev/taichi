#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"

TC_NAMESPACE_BEGIN

namespace Tlang {
// TODO: do we need polymorphism here?
class Node {
public:
  enum class Type : int { mul, add, sub, div, load, store, combine, constant };

  Address addr;
  std::vector<Expr> ch;       // Four child max
  std::vector<Expr> members;  // for vectorized instructions
  Type type;
  std::string var_name;
  float64 value;
  bool is_vectorized;

  Node(Type type) : type(type) {
    is_vectorized = false;
  }

  Node(Type type, Expr ch0, Expr ch1);

  int member_id(const Expr &expr) const;
};

using NodeType = Node::Type;

// Reference counted...
class Expr {
private:
  Handle<Node> node;

public:
  Expr() {
  }

  Expr(float64 val) {
    // create a constant node
    node = std::make_shared<Node>(NodeType::constant);
    node->value = val;
  }

  Expr(Handle<Node> node) : node(node) {
  }

  template <typename... Args>
  static Expr create(Args &&... args) {
    return Expr(std::make_shared<Node>(std::forward<Args>(args)...));
  }

#define BINARY_OP(op, name)                            \
  Expr operator op(const Expr &o) const {              \
    return Expr::create(NodeType::name, node, o.node); \
  }

  BINARY_OP(*, mul);
  BINARY_OP(+, add);
  BINARY_OP(-, sub);
  BINARY_OP(/, div);
#undef BINARY_OP

  Expr store(const Expr &e) {
    if (!node) {
      node = std::make_shared<Node>(NodeType::combine);
    }
    auto n = std::make_shared<Node>(NodeType::store);
    n->ch.push_back(e.node);
    Expr store_e(n);
    node->ch.push_back(n);
    return store_e;
  }

  Expr store(const Expr &e, Address addr) {
    if (!node) {
      node = std::make_shared<Node>(NodeType::combine);
    }
    auto n = std::make_shared<Node>(NodeType::store);
    n->ch.push_back(e.node);
    n->addr = addr;
    Expr store_e(n);
    node->ch.push_back(n);
    return store_e;
  }

  Node *operator->() {
    return node.get();
  }

  const Node *operator->() const {
    return node.get();
  }

  bool operator<(const Expr &o) const {
    return node.get() < o.node.get();
  }

  operator bool() const {
    return node.get() != nullptr;
  }

  operator void *() const {
    return (void *)node.get();
  }

  bool operator==(const Expr &o) const {
    return (void *)(*this) == (void *)o;
  }

  void accept(Visitor &visitor) {
    if (visitor.order == Visitor::Order::parent_first) {
      visitor.visit(*this);
    }
    for (auto &c : this->node->ch) {
      c.accept(visitor);
    }
    if (visitor.order == Visitor::Order::child_first) {
      visitor.visit(*this);
    }
  }
};

inline bool prior_to(Expr &a, Expr &b) {
  auto address1 = a->addr;
  auto address2 = b->addr;
  return address1.same_type(address2) &&
         address1.offset() + 1 == address2.offset();
}

inline Node::Node(Type type, Expr ch0, Expr ch1) : Node(type) {
  ch.resize(2);
  ch[0] = ch0;
  ch[1] = ch1;
}

inline Expr placeholder() {
  auto n = std::make_shared<Node>(NodeType::load);
  return Expr(n);
}

inline Expr load(Address addr) {
  auto n = std::make_shared<Node>(NodeType::load);
  TC_ASSERT(addr.initialized());
  n->addr = addr;
  TC_ASSERT(0 <= addr.buffer_id && addr.buffer_id < 3);
  return Expr(n);
}

inline AddrNode &AddrNode::place(Expr &expr) {
  if (!expr) {
    expr = placeholder();
  }
  TC_ASSERT(depth >= 3);
  TC_ASSERT(this->addr == nullptr);
  ch.push_back(create(depth + 1, &expr->addr));
  return *this;
}

inline int Node::member_id(const Expr &expr) const {
  for (int i = 0; i < (int)members.size(); i++) {
    if (members[i] == expr) {
      return i;
    }
  }
  return -1;
}

}

TC_NAMESPACE_END
