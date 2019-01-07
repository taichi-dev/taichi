#pragma once

#include "util.h"
#include "address.h"

TLANG_NAMESPACE_BEGIN

class SNode;

class Node {
 private:
  static int counter;

 public:
  static constexpr int num_additional_values = 8;
  static void reset_counter() {
    counter = 0;
  }

  // When adding members variables, make sure to modify copy_to.

  std::vector<Expr> ch;
  std::vector<Expr> members;  // for vectorized instructions
  NodeType type;
  DataType data_type;
  BinaryType binary_type;
  std::string var_name;
  std::vector<uint64> attributes[num_additional_values];
  int lanes;
  int id;
  int num_groups_;
  std::string name_;

  void copy_to(Node &node) {
    node.ch = ch;
    node.members = members;
    node.type = type;
    node.data_type = data_type;
    node.binary_type = binary_type;
    node.var_name = var_name;
    for (int i = 0; i < num_additional_values; i++) {
      node.attributes[i] = attributes[i];
    }
    node.lanes = lanes;
    node.num_groups_ = num_groups_;
  }

  Node(const Node &) = delete;

  Node(NodeType type) : type(type) {
    data_type = DataType::f32;
    binary_type = BinaryType::undefined;
    id = counter++;
    this->lanes = 0;
    set_lanes(1);
  }

  void set_lanes(int lanes) {
    for (int i = 0; i < num_additional_values; i++) {
      attributes[i].resize(lanes);
      if (lanes > this->lanes) {
        std::fill(attributes[i].begin() + this->lanes, attributes[i].end(), 0);
      }
    }
    this->lanes = lanes;
  }

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

  // TODO: remove this
  int vv_width() {
    return group_size() * num_groups();
  }

  std::string data_type_name() const {
    return taichi::Tlang::data_type_name(data_type);
  }

  std::string node_type_name() const {
    return taichi::Tlang::node_type_name(type);
  }

  Node(NodeType type, Expr ch0);

  Node(NodeType type, Expr ch0, Expr ch1);

  Node(NodeType type, Expr ch0, Expr ch1, Expr ch2);

  int member_id(const Expr &expr) const;

  template <typename T>
  T &value(int i = 0) {
    return attribute<T>(0, i);
  }

  template <typename T = uint64>
  T &attribute(int k, int i = 0) {
    return *reinterpret_cast<T *>(&attributes[k][i]);
  }

  int &index_id(int i) {
    TC_ASSERT(type == NodeType::index);
    return attribute<int>(0, i);
  }

  int &index_offset(int i) {
    TC_ASSERT(type == NodeType::index);
    return attribute<int>(1, i);
  }

  SNode *&new_addresses(int i) {
    TC_ASSERT(type == NodeType::addr || type == NodeType::touch);
    return attribute<SNode *>(0, i);
  }

  void set_similar(const Expr &expr);
};

TLANG_NAMESPACE_END
