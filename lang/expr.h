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
  // TODO: rename
  enum class Type : int {
    mul,
    add,
    sub,
    div,
    load,
    store,
    pointer,
    combine,
    index,
    addr,
    cache_store,  // -> adapter
    cache_load,
    imm
  };

  using NodeType = Type;

  enum class DataType : int {
    f16,
    f32,
    f64,
    i8,
    i16,
    i32,
    i64,
    u8,
    u16,
    u32,
    u64
  };

  std::vector<Expr> ch;       // Four child max
  std::vector<Expr> members;  // for vectorized instructions
  Type type;
  DataType data_type;
  std::string var_name;
  float64 _value;
  bool is_vectorized;
  static std::map<DataType, std::string> data_type_names;
  static std::map<Type, std::string> node_type_names;

  Node(Type type) : type(type) {
    is_vectorized = false;
    data_type = DataType::f32;
  }

  std::string data_type_name() {
    if (data_type_names.empty()) {
#define REGISTER_DATA_TYPE(i) data_type_names[DataType::i] = #i;
      REGISTER_DATA_TYPE(f16);
      REGISTER_DATA_TYPE(f32);
      REGISTER_DATA_TYPE(f64);
      REGISTER_DATA_TYPE(i8);
      REGISTER_DATA_TYPE(i16);
      REGISTER_DATA_TYPE(i32);
      REGISTER_DATA_TYPE(i64);
      REGISTER_DATA_TYPE(u8);
      REGISTER_DATA_TYPE(u16);
      REGISTER_DATA_TYPE(u32);
      REGISTER_DATA_TYPE(u64);
    }
    return data_type_names[data_type];
  }

  std::string node_type_name() {
    if (node_type_names.empty()) {
#define REGISTER_NODE_TYPE(i) node_type_names[NodeType::i] = #i;
      REGISTER_NODE_TYPE(mul);
      REGISTER_NODE_TYPE(add);
      REGISTER_NODE_TYPE(sub);
      REGISTER_NODE_TYPE(div);
      REGISTER_NODE_TYPE(load);
      REGISTER_NODE_TYPE(store);
      REGISTER_NODE_TYPE(combine);
      REGISTER_NODE_TYPE(addr);
      REGISTER_NODE_TYPE(pointer);
      REGISTER_NODE_TYPE(cache_store);
      REGISTER_NODE_TYPE(cache_load);
      REGISTER_NODE_TYPE(imm);
      REGISTER_NODE_TYPE(index);
    }
    return node_type_names[type];
  }

  Address &get_address_() {  // TODO: remove this hack
    return _addr;
  }

  Address &get_address() {
    TC_ASSERT(type == Type::addr);
    TC_ERROR_UNLESS(ch.size() == 1,
                    "Should have exactly one index child, instead of {}",
                    ch.size());
    return _addr;
  }

  Address &addr();

  Node(Type type, Expr ch0);

  Node(Type type, Expr ch0, Expr ch1);

  int member_id(const Expr &expr) const;

  template <typename T>
  T &value() {
    return *reinterpret_cast<T *>(&_value);
  }
};

using NodeType = Node::Type;

class Visitor;

// Reference counted...
class Expr {
 private:
  Handle<Node> node;

 public:
  using Type = Node::Type;

  auto &get_node() {
    return node;
  }

  Expr() {
  }

  Expr(float64 val) {
    // create a constant node
    node = std::make_shared<Node>(NodeType::imm);
    node->value<float64>() = val;
  }

  Expr(Handle<Node> node) : node(node) {
  }

  template <typename... Args>
  static Expr create(Args &&... args) {
    return Expr(std::make_shared<Node>(std::forward<Args>(args)...));
  }

  template <typename T>
  static Expr create_imm(T t) {
    auto e = create(Type::imm);
    e->value<T>() = t;
    return e;
  }

  static Expr index(int i) {
    auto e = create(Type::index);
    e->value<int>() = i;
    return e;
  }

  static Expr load_if_pointer(const Expr &in) {
    if (in->type == NodeType::pointer) {
      return create(NodeType::load, in);
    } else {
      return in;
    }
  }

#define BINARY_OP(op, name)                                    \
  Expr operator op(const Expr &o) const {                      \
    return Expr::create(NodeType::name, load_if_pointer(node), \
                        load_if_pointer(o.node));              \
  }

  BINARY_OP(*, mul);
  BINARY_OP(+, add);
  BINARY_OP(-, sub);
  BINARY_OP(/, div);
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

  Expr operator[](const Expr &i);
};

using Index = Expr;

inline bool prior_to(Address address1, Address address2) {
  return address1.same_type(address2) &&
         address1.offset() + 1 == address2.offset();
}

inline bool prior_to(Expr &a, Expr &b) {
  TC_ASSERT(a->type == NodeType::pointer && b->type == NodeType::pointer);
  return prior_to(a->ch[0]->get_address(), b->ch[0]->get_address());
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
  TC_ASSERT(ch[0]->type == Type::pointer);
  return ch[0]->ch[0]->get_address();
}

inline Expr load(const Expr &addr) {
  auto expr = Expr::create(NodeType::load);
  expr->ch.push_back(addr);
  return expr;
}
}

TC_NAMESPACE_END
