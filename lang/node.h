#pragma once

#include "util.h"
#include "address.h"

TLANG_NAMESPACE_BEGIN

class SNode;

class Node {
 private:
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

  static constexpr int num_additional_values = 8;
  std::vector<float64> attributes[num_additional_values];
  int lanes;
  int id;
  int num_groups_;
  bool is_vectorized;
  std::string name_;

  Node(const Node &) = delete;

  Node(NodeType type) : type(type) {
    is_vectorized = false;
    data_type = DataType::f32;
    binary_type = BinaryType::undefined;
    id = counter++;
    set_lanes(1);
  }

  // erases all data
  void set_lanes(int lanes) {
    this->lanes = lanes;
    for (int i = 0; i < num_additional_values; i++) {
      attributes[i].resize(lanes);
      std::fill(attributes[i].begin(), attributes[i].end(), 0.0_f64);
    }
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

  template <typename T>
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
    TC_ASSERT(type == NodeType::addr);
    return attribute<SNode *>(0, i);
  }

  void set_similar(const Expr &expr);
};

TLANG_NAMESPACE_END
