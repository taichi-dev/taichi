#pragma once

#include "util.h"
#include "address.h"

TLANG_NAMESPACE_BEGIN

// Regular binary ops:
// Operations that take two oprands, and returns a single operand with the same
// type

enum class BinaryType : int { mul, add, sub, div, mod, max, min, undefined };

extern std::map<BinaryType, std::string> binary_type_names;

inline std::string binary_type_name(BinaryType type) {
  if (binary_type_names.empty()) {
#define REGISTER_BINARY_TYPE(i) binary_type_names[BinaryType::i] = #i;
    REGISTER_BINARY_TYPE(mul);
    REGISTER_BINARY_TYPE(add);
    REGISTER_BINARY_TYPE(sub);
    REGISTER_BINARY_TYPE(div);
    REGISTER_BINARY_TYPE(mod);
    REGISTER_BINARY_TYPE(max);
    REGISTER_BINARY_TYPE(min);
  }
  return binary_type_names[type];
}

enum class NodeType : int {
  binary,  // regular binary
  land,
  load,
  store,
  pointer,
  combine,
  index,
  addr,
  adapter_store,
  adapter_load,
  imm,
  floor,
  cast,
  shr,
  shl,
  cmp,
  select,
  undefined
};

extern std::map<NodeType, std::string> node_type_names;

inline std::string node_type_name(NodeType type) {
  if (node_type_names.empty()) {
#define REGISTER_NODE_TYPE(i) node_type_names[NodeType::i] = #i;
    REGISTER_NODE_TYPE(binary);
    REGISTER_NODE_TYPE(land);
    REGISTER_NODE_TYPE(load);
    REGISTER_NODE_TYPE(store);
    REGISTER_NODE_TYPE(combine);
    REGISTER_NODE_TYPE(addr);
    REGISTER_NODE_TYPE(pointer);
    REGISTER_NODE_TYPE(adapter_store);
    REGISTER_NODE_TYPE(adapter_load);
    REGISTER_NODE_TYPE(imm);
    REGISTER_NODE_TYPE(index);
    REGISTER_NODE_TYPE(floor);
    REGISTER_NODE_TYPE(cast);
    REGISTER_NODE_TYPE(shr);
    REGISTER_NODE_TYPE(shl);
    REGISTER_NODE_TYPE(cmp);
    REGISTER_NODE_TYPE(select);
  }
  return node_type_names[type];
}

enum class CmpType { eq, ne, le, lt };

class SNode;

class Node {
 private:
  Address _addr;
  static int counter;

 public:
  static void reset_counter() {
    counter = 0;
  }

  std::vector<Expr> ch;       // Four child max
  std::vector<Expr> members;  // for vectorized instructions
  NodeType type;
  DataType data_type;
  BinaryType binary_type;
  std::string var_name;
  float64 _value;
  int id;
  int num_groups_;
  bool is_vectorized;
  std::string name_;
  SNode *new_address;

  std::string name() {
    return name_;
  }

  void name(std::string s) {
    name_ = s;
  }

  int group_size() const;

  int &num_groups() {
    return num_groups_;
  }

  int vv_width() {
    return group_size() * num_groups();
  }

  Node(const Node &) = delete;

  Node(NodeType type) : type(type) {
    is_vectorized = false;
    data_type = DataType::f32;
    binary_type = BinaryType::undefined;
    id = counter++;
    _value = 0;
    new_address = nullptr;
  }

  std::string data_type_name() const {
    return taichi::Tlang::data_type_name(data_type);
  }

  std::string node_type_name() const {
    return taichi::Tlang::node_type_name(type);
  }

  Address &get_address_() {  // TODO: remove this hack
    return _addr;
  }

  Address &get_address() {
    TC_ASSERT(type == NodeType::addr);
    return _addr;
  }

  Address &addr();

  Node(NodeType type, Expr ch0);

  Node(NodeType type, Expr ch0, Expr ch1);

  Node(NodeType type, Expr ch0, Expr ch1, Expr ch2);

  int member_id(const Expr &expr) const;

  template <typename T>
  T &value() {
    return *reinterpret_cast<T *>(&_value);
  }
};

TLANG_NAMESPACE_END
