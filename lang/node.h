#pragma once

#include "util.h"
#include "address.h"

TLANG_NAMESPACE_BEGIN


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
