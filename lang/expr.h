#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"
#include "address.h"
#include "visitor.h"
#include "node.h"

TLANG_NAMESPACE_BEGIN

class Visitor;

// Reference counted...
class Expr {
 private:
  Handle<Node> node;
  static bool allow_store;

 public:
  static void set_allow_store(bool val) {
    allow_store = val;
  }

  auto &get_node() {
    return node;
  }

  Expr() {
  }

  /*
  Expr(float64 val) {
    // create a constant node
    node = std::make_shared<Node>(NodeType::imm);
    node->value<float64>() = val;
  }
  */

  Expr(Handle<Node> node) : node(node) {
  }

  template <typename... Args>
  static Expr create(Args &&... args) {
    return Expr(std::make_shared<Node>(std::forward<Args>(args)...));
  }

  template <typename T>
  static Expr create_imm(T t) {
    auto e = create(NodeType::imm);
    e->value<T>() = t;
    return e;
  }

  static Expr index(int i) {
    auto e = create(NodeType::index);
    e->value<int>() = i;
    e->data_type = DataType::i32;
    return e;
  }

  static Expr load_if_pointer(const Expr &in) {
    if (in->type == NodeType::pointer) {
      return create(NodeType::load, in);
    } else {
      return in;
    }
  }

#define REGULAR_BINARY_OP(op, name)                                \
  Expr operator op(const Expr &o) const {                          \
    TC_ASSERT(node->data_type == o->data_type)                     \
    auto t = Expr::create(NodeType::binary, load_if_pointer(node), \
                          load_if_pointer(o.node));                \
    t->data_type = o->data_type;                                   \
    t->binary_type = BinaryType::name;                             \
    return t;                                                      \
  }

#define BINARY_OP(op, name)                                      \
  Expr operator op(const Expr &o) const {                        \
    TC_ASSERT(node->data_type == o->data_type)                   \
    auto t = Expr::create(NodeType::name, load_if_pointer(node), \
                          load_if_pointer(o.node));              \
    t->data_type = o->data_type;                                 \
    return t;                                                    \
  }

  REGULAR_BINARY_OP(*, mul);
  REGULAR_BINARY_OP(+, add);
  REGULAR_BINARY_OP(-, sub);
  REGULAR_BINARY_OP(/, div);
  REGULAR_BINARY_OP(%, mod);
  BINARY_OP(&, land);
  BINARY_OP(>>, shr);
  BINARY_OP(<<, shl);
#undef BINARY_OP

  // ch[0] = address
  // ch[1] = data
  Expr store(const Expr &pointer, const Expr &e) {
    if (!node) {
      node = std::make_shared<Node>(NodeType::combine);
    }
    auto n = std::make_shared<Node>(NodeType::store);
    TC_ASSERT(pointer->type == NodeType::pointer);
    n->ch.push_back(pointer);
    n->ch.push_back(e);
    Expr store_e(n);
    node->ch.push_back(n);
    return store_e;
  }

  Node *operator->() {
    TC_ASSERT(node.get() != nullptr);
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

  bool operator!=(const Expr &o) const {
    return (void *)(*this) != (void *)o;
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

  Expr operator[](const Expr &i);

  Expr &operator[](int i) {
    TC_ASSERT(0 <= i && i < (int)node->ch.size());
    return node->ch[i];
  }

  void set(const Expr &o) {
    node = o.node;
  }

  Expr &name(std::string s) {
    node->name(s);
    return *this;
  }

  /*
  Expr operator!=(const Expr &o) {
    auto n = create(NodeType::cmp, *this, o);
    n->value<CmpType>() = CmpType::ne;
    return n;
  }

  Expr operator<(const Expr &o) {
    auto n = create(NodeType::cmp, *this, o);
    n->value<CmpType>() = CmpType::lt;
    return n;
  }
  */

  Node *ptr() {
    return node.get();
  }

  void *evaluate_addr(int i);

  template <typename T>
  void set(int i, T t) {
    TC_ASSERT(get_data_type<T>() == node->data_type);
    *(T *)evaluate_addr(i) = t;
  }

  template <typename T>
  T get(int i) {
    TC_ASSERT(get_data_type<T>() == node->data_type);
    return *(T *)evaluate_addr(i);
  }
};

using Index = Expr;

inline Expr cmp_ne(const Expr &a, const Expr &b) {
  auto n = Expr::create(NodeType::cmp, a, b);
  n->value<CmpType>() = CmpType::ne;
  return n;
}

inline Expr cmp_lt(const Expr &a, const Expr &b) {
  auto n = Expr::create(NodeType::cmp, a, b);
  n->value<CmpType>() = CmpType::lt;
  return n;
}

inline bool prior_to(Address address1, Address address2) {
  return address1.same_type(address2) &&
         address1.offset() + 1 == address2.offset();
}

inline bool prior_to(Expr &a, Expr &b) {
  TC_ASSERT(a->type == NodeType::pointer && b->type == NodeType::pointer);
  return prior_to(a->ch[0]->get_address(), b->ch[0]->get_address());
}

inline Node::Node(NodeType type, Expr ch0) : Node(type) {
  ch.resize(1);
  ch[0] = ch0;
}

inline Node::Node(NodeType type, Expr ch0, Expr ch1) : Node(type) {
  ch.resize(2);
  ch[0] = ch0;
  ch[1] = ch1;
}

inline Node::Node(NodeType type, Expr ch0, Expr ch1, Expr ch2) : Node(type) {
  ch.resize(3);
  ch[0] = ch0;
  ch[1] = ch1;
  ch[2] = ch2;
}

inline Expr placeholder(DataType dt) {
  auto n = std::make_shared<Node>(NodeType::addr);
  n->data_type = dt;
  return Expr(n);
}

inline Expr variable(DataType dt) {
  return placeholder(dt);
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
  TC_ASSERT(type == NodeType::load || type == NodeType::store);
  TC_ASSERT(ch.size());
  TC_ASSERT(ch[0]->type == NodeType::pointer);
  return ch[0]->ch[0]->get_address();
}

inline Expr load(const Expr &addr) {
  auto expr = Expr::create(NodeType::load);
  expr->ch.push_back(addr);
  return expr;
}

inline Expr select(Expr mask, Expr true_val, Expr false_val) {
  return Expr::create(NodeType::select, mask, true_val, false_val);
}

TLANG_NAMESPACE_END
