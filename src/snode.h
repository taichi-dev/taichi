#pragma once
#include "util.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

class Expr;

TC_FORCE_INLINE int32 constexpr operator"" _bits(unsigned long long a) {
  return 1 << a;
}

struct IndexExtractor {
  int start, num_bits;  //, dest_offset;
  int acc_offset;
  int dimension;

  TC_IO_DEF(start, num_bits, acc_offset);

  // TODO: rename start to src_offset

  bool active;

  IndexExtractor() {
    start = 0;
    num_bits = 0;
    // dest_offset = 0;
    active = false;
    dimension = 1;
    acc_offset = 0;
  }

  void activate(int num_bits) {
    active = true;
    this->num_bits = num_bits;
    dimension = 1 << num_bits;
  }
};

struct Matrix;
class Expr;

class Index {
 public:
  int value;
  Index() {
    value = 0;
  }
  Index(int value) : value(value) {
    TC_ASSERT(0 <= value && value < max_num_indices);
  }
};

// "Structural" nodes
class SNode {
 public:
  std::vector<Handle<SNode>> ch;

  IndexExtractor extractors[max_num_indices];
  int taken_bits[max_num_indices];  // counting from the tail
  int num_active_indices;
  int physical_index_position[max_num_indices];
  // physical_index_position[i] =
  // the virtual index position of the i^th physical index

  static int counter;
  int id;
  int depth;
  bool _verbose;
  bool _multi_threaded;

  std::string name;
  int64 n;
  int total_num_bits, total_bit_start;
  DataType dt;
  bool has_ambient;
  TypedConstant ambient_val;
  // Note: parent will not be set until structural nodes are compiled!
  SNode *parent;

  std::string data_type_name() {
    return Tlang::data_type_name(dt);
  }

  using AccessorFunction = void *(*)(void *, int, int, int, int);
  using StatFunction = AllocatorStat (*)();
  AccessorFunction access_func;
  StatFunction stat_func;

  std::string node_type_name;
  SNodeType type;
  int index_id;

  SNode() {
    id = counter++;
  }

  SNode(int depth, SNodeType t) : depth(depth), type(t) {
    id = counter++;
    total_num_bits = 0;
    total_bit_start = 0;
    num_active_indices = 0;
    std::memset(taken_bits, 0, sizeof(taken_bits));
    std::memset(physical_index_position, -1, sizeof(physical_index_position));
    access_func = nullptr;
    stat_func = nullptr;
    parent = nullptr;
    _verbose = false;
    _multi_threaded = false;
    index_id = -1;
    has_ambient = false;
    dt = DataType::unknown;
  }

  SNode &insert_children(SNodeType t) {
    ch.push_back(create(depth + 1, t));
    // Note: parent will not be set until structural nodes are compiled!
    return *ch.back();
  }

  SNode &create_node(std::vector<Index> indices,
                     std::vector<int> sizes,
                     SNodeType type);

  // SNodes maintains how flattened index bits are taken from indices
  SNode &dense(std::vector<Index> indices, std::vector<int> sizes) {
    return create_node(indices, sizes, SNodeType::dense);
  }

  SNode &dense(std::vector<Index> indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::dense);
  }

  SNode &dense(const Index &index, int size) {
    return SNode::dense(std::vector<Index>{index}, size);
  }

  SNode &multi_threaded(bool val = true) {
    this->_multi_threaded = val;
    return *this;
  }

  SNode &verbose() {
    this->_verbose = true;
    return *this;
  }

  template <typename... Args>
  SNode &place(Expr &expr, Args &&... args) {
    return place(expr).place(std::forward<Args>(args)...);
  }

  SNode &place(Matrix &mat);

  template <typename... Args>
  static Handle<SNode> create(Args &&... args) {
    return std::make_shared<SNode>(std::forward<Args>(args)...);
  }

  std::string type_name() {
    return snode_type_name(type);
  }

  void print() {
    for (int i = 0; i < depth; i++) {
      fmt::print("  ");
    }
    fmt::print("{}\n", type_name());
    for (auto c : ch) {
      c->print();
    }
  }

  SNode &place(Expr &expr);

  /*
  SNode &place_verbose(Expr &expr) {
    place(expr);
    ch.back()->verbose();
    return *this;
  }
  */

  SNode &indirect(Index &expr, int n) {
    auto &child = insert_children(SNodeType::indirect);
    child.index_id = expr.value;
    child.n = n;
    return child;
  }

  SNode &dynamic(Index &expr, int n) {
    TC_ASSERT(bit::is_power_of_two(n));
    auto &child = insert_children(SNodeType::dynamic);
    child.extractors[expr.value].activate(bit::log2int(n));
    child.n = n;
    return child;
  }

  SNode &hashed(std::vector<Index> indices, std::vector<int> sizes) {
    return create_node(indices, sizes, SNodeType::hashed);
  }

  SNode &hashed(std::vector<Index> indices, int sizes) {
    return create_node(indices, std::vector<int>{sizes}, SNodeType::hashed);
  }

  SNode &hashed(Index &expr, int n) {
    return hashed(std::vector<Index>{expr}, n);
  }

  SNode &pointer() {
    return insert_children(SNodeType::pointer);
  }

  TC_FORCE_INLINE void *evaluate(void *ds, int i, int j, int k, int l) {
    TC_ASSERT(access_func);
    return access_func(ds, i, j, k, l);
  }

  TC_FORCE_INLINE AllocatorStat stat() {
    TC_ASSERT(stat_func);
    return stat_func();
  }

  int child_id(SNode *c) {
    for (int i = 0; i < (int)ch.size(); i++) {
      if (ch[i].get() == c) {
        return i;
      }
    }
    return -1;
  }
};

TLANG_NAMESPACE_END
