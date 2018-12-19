#pragma once

#include <taichi/common/util.h>
#include "../headers/common.h"
#include "address.h"
#include "visitor.h"

TC_NAMESPACE_BEGIN

namespace Tlang {

struct Index {

};

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
    pointer,
    store,
    combine,
    addr,
    cache_store,
    cache_load,
    imm
  };

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

  std::string type_name() {
    if (type == Type::mul) {
      return "mul";
    } else if (type == Type::add) {
      return "add";
    } else if (type == Type::sub) {
      return "sub";
    } else if (type == Type::div) {
      return "div";
    } else if (type == Type::load) {
      return "load";
    } else if (type == Type::store) {
      return "store";
    } else if (type == Type::combine) {
      return "combine";
    } else if (type == Type::addr) {
      return "addr";
    } else if (type == Type::pointer) {
      return "pointer";
    } else if (type == Type::combine) {
      return "addr";
    } else if (type == Type::cache_store) {
      return "cache_store";
    } else if (type == Type::cache_load) {
      return "cache_load";
    } else if (type == Type::imm) {
      return "imm";
    } else {
      TC_NOT_IMPLEMENTED;
    }
    return "X";
  }

  Address &get_address() {
    TC_ASSERT(type == Type::addr);
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

  Expr operator[](Index i);
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
