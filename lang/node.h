#pragma once

#include "util.h"
#include "address.h"

TLANG_NAMESPACE_BEGIN

class Node {
private:
  Address _addr;
  static int counter;

public:
  static void reset_counter() {
    counter = 0;
  }

  // TODO: rename
  enum class Type : int {
    mul,
    add,
    sub,
    div,
    mod,
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
    max,
    min,
    cast,
    land,
    shr,
    shl,
    cmp,
    select,
  };

  enum class CmpType { eq, ne, le, lt };

  using NodeType = Type;

  std::vector<Expr> ch;       // Four child max
  std::vector<Expr> members;  // for vectorized instructions
  Type type;
  DataType data_type;
  std::string var_name;
  float64 _value;
  int id;
  int num_groups_;
  bool is_vectorized;
  static std::map<Type, std::string> node_type_names;
  std::string name_;

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

  Node(Type type) : type(type) {
    is_vectorized = false;
    data_type = DataType::f32;
    id = counter++;
    _value = 0;
  }

  std::string data_type_name() {
    return taichi::Tlang::data_type_name(data_type);
  }

  std::string node_type_name() {
    if (node_type_names.empty()) {
#define REGISTER_NODE_TYPE(i) node_type_names[NodeType::i] = #i;
      REGISTER_NODE_TYPE(mul);
      REGISTER_NODE_TYPE(add);
      REGISTER_NODE_TYPE(sub);
      REGISTER_NODE_TYPE(div);
      REGISTER_NODE_TYPE(mod);
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
      REGISTER_NODE_TYPE(max);
      REGISTER_NODE_TYPE(min);
      REGISTER_NODE_TYPE(cast);
      REGISTER_NODE_TYPE(land);
      REGISTER_NODE_TYPE(shr);
      REGISTER_NODE_TYPE(shl);
      REGISTER_NODE_TYPE(cmp);
      REGISTER_NODE_TYPE(select);
    }
    return node_type_names[type];
  }

  Address &get_address_() {  // TODO: remove this hack
    return _addr;
  }

  Address &get_address() {
    TC_ASSERT(type == Type::addr);
    return _addr;
  }

  Address &addr();

  Node(Type type, Expr ch0);

  Node(Type type, Expr ch0, Expr ch1);

  Node(Type type, Expr ch0, Expr ch1, Expr ch2);

  int member_id(const Expr &expr) const;

  template <typename T>
  T &value() {
    return *reinterpret_cast<T *>(&_value);
  }
};

TLANG_NAMESPACE_END
