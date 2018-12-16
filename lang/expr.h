#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"
#include "address.h"
#include "visitor.h"

TC_NAMESPACE_BEGIN

namespace Tlang {
// TODO: do we need polymorphism here?
class Node {
 private:
  Address _addr;

 public:
  enum class Type : int {
    mul,
    add,
    sub,
    div,
    load,
    store,
    combine,
    constant,
    addr
  };

  std::vector<Expr> ch;       // Four child max
  std::vector<Expr> members;  // for vectorized instructions
  Type type;
  std::string var_name;
  float64 value;
  bool is_vectorized;

  Node(Type type) : type(type) {
    is_vectorized = false;
  }

  Address &get_address() {
    TC_ASSERT(type == Type::addr);
    return _addr;
  }

  Address &addr();

  Node(Type type, Expr ch0);

  Node(Type type, Expr ch0, Expr ch1);

  int member_id(const Expr &expr) const;
};

using NodeType = Node::Type;

class Visitor;

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

  static Expr load_if_addr(const Expr &in) {
    if (in->type == NodeType::addr) {
      return create(NodeType::load, in);
    } else {
      return in;
    }
  }

#define BINARY_OP(op, name)                                 \
  Expr operator op(const Expr &o) const {                   \
    return Expr::create(NodeType::name, load_if_addr(node), \
                        load_if_addr(o.node));              \
  }

  BINARY_OP(*, mul);
  BINARY_OP(+, add);
  BINARY_OP(-, sub);
  BINARY_OP(/, div);
#undef BINARY_OP

  // ch[0] = address
  // ch[1] = data
  Expr store(const Expr &addr, const Expr &e) {
    if (!node) {
      node = std::make_shared<Node>(NodeType::combine);
    }
    auto n = std::make_shared<Node>(NodeType::store);
    n->ch.push_back(addr);
    n->ch.push_back(e.node);
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

  Expr &operator=(const Expr &o);
};

inline bool prior_to(Address address1, Address address2) {
  return address1.same_type(address2) &&
         address1.offset() + 1 == address2.offset();
}

inline bool prior_to(Expr &a, Expr &b) {
  return prior_to(a->get_address(), b->get_address());
}

inline Node::Node(Type type, Expr ch0) : Node(type) {
  ch.resize(1);
  ch[0] = ch0;
}

inline Node::Node(Type type, Expr ch0, Expr ch1) : Node(type) {
  ch.resize(2);
  ch[0] = ch0;
  ch[1] = ch1;
}

inline Expr placeholder() {
  auto n = std::make_shared<Node>(NodeType::addr);
  return Expr(n);
}

inline int Node::member_id(const Expr &expr) const {
  for (int i = 0; i < (int)members.size(); i++) {
    if (members[i] == expr) {
      return i;
    }
  }
  return -1;
}

inline Address &Node::addr() {
  TC_ASSERT(type == Type::load || type == Type::store);
  TC_ASSERT(ch.size());
  return ch[0]->get_address();
}

inline Expr load(const Expr &addr) {
  auto expr = Expr::create(NodeType::load);
  expr->ch.push_back(addr);
  return expr;
}
}

TC_NAMESPACE_END
