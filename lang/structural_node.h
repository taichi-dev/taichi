#pragma once
#include "expr.h"
#include "ir.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

TC_FORCE_INLINE int32 constexpr operator"" _bits(unsigned long long a) {
  return 1 << a;
}

struct IndexExtractor {
  int start, num_bits, dest_offset;

  TC_IO_DEF(start, num_bits, dest_offset);

  // TODO: rename start to src_offset

  IndexExtractor() {
    start = 0;
    num_bits = 0;
    dest_offset = 0;
  }
};

struct Matrix;
class ExpressionHandle;

// "Structural" nodes
class SNode {
 public:
  std::vector<Handle<SNode>> ch;

  IndexExtractor extractors[max_num_indices];
  int taken_bits[max_num_indices];  // counting from the tail
  int num_active_indices;
  int index_order[max_num_indices];  // look_up(index[index_order[index_id]]);

  static int counter;
  int id;
  int depth;
  bool _verbose;
  bool _multi_threaded;

  std::string name;
  int64 n;
  int total_num_bits, total_bit_start;
  DataType dt;
  // Expr addr;
  // Note: parent will not be set until structural nodes are compiled!
  SNode *parent;

  std::string data_type_name() {
    return Tlang::data_type_name(dt);
  }

  using AccessorFunction = void *(*)(void *, int, int, int, int);
  AccessorFunction func;

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
    std::memset(index_order, -1, sizeof(index_order));
    func = nullptr;
    parent = nullptr;
    _verbose = false;
    _multi_threaded = false;
    index_id = -1;
  }

  SNode &insert_children(SNodeType t) {
    ch.push_back(create(depth + 1, t));
    // Note: parent will not be set until structural nodes are compiled!
    return *ch.back();
  }

  // Let us deal with 1D case first
  // SNodes maintains how flattened index bits are taken from indices
  SNode &fixed(std::vector<Expr> indices, std::vector<int> sizes) {
    TC_ASSERT(indices.size() == sizes.size())
    bool all_one = true;
    for (auto s : sizes) {
      if (s != 1) {
        all_one = false;
      }
    }
    if (all_one)
      return *this;  // do nothing
    auto &new_node = insert_children(SNodeType::fixed);
    new_node.n = 1;
    for (auto s : sizes) {
      TC_ASSERT(bit::is_power_of_two(s));
      new_node.n *= s;
    }
    for (int i = 0; i < (int)indices.size(); i++) {
      auto &ind = indices[i];
      TC_ASSERT(ind->lanes == 1);
      new_node.extractors[ind->index_id(0)].num_bits = bit::log2int(sizes[i]);
    }
    return new_node;
  }

  SNode &fixed(const Expr &index, int size) {
    return SNode::fixed(std::vector<Expr>{index}, {size});
  }

  SNode &forked() {
    auto &new_node = insert_children(SNodeType::forked);
    return new_node;
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

  SNode &place(Expr &expr) {
    auto &child = insert_children(SNodeType::place);
    expr->snode_ptr(0) = &child;
    // child.addr.set(expr);
    return *this;
  }

  SNode &place_new(ExpressionHandle &expr);

  SNode &place_verbose(Expr &expr) {
    place(expr);
    ch.back()->verbose();
    return *this;
  }

  SNode &indirect(Expr &expr, int n) {
    auto &child = insert_children(SNodeType::indirect);
    TC_ASSERT(expr->type == NodeType::index);
    child.index_id = expr->value<int>(0);
    // child.extractors[child.index_id].num_bits = 0;
    child.n = n;
    return child;
  }

  SNode &dynamic(Expr &expr, int n) {
    TC_ASSERT(bit::is_power_of_two(n));
    auto &child = insert_children(SNodeType::dynamic);
    TC_ASSERT(expr->type == NodeType::index);
    // child.index_id = expr->value<int>(0);
    child.extractors[expr->value<int>(0)].num_bits = bit::log2int(n);
    child.n = n;
    return child;
  }

  SNode &hashed(Expr &expr, int n) {
    TC_ASSERT_INFO(depth == 0,
                   "hashed node must be child of root due to initialization "
                   "memset limitation.");
    TC_ASSERT(bit::is_power_of_two(n));
    auto &child = insert_children(SNodeType::hashed);
    TC_ASSERT(expr->type == NodeType::index);
    // child.index_id = expr->value<int>(0);
    child.extractors[expr->value<int>(0)].num_bits = bit::log2int(n);
    child.n = n;
    return child;
  }

  SNode &pointer() {
    return insert_children(SNodeType::pointer);
  }

  TC_FORCE_INLINE void *evaluate(void *ds, int i, int j, int k, int l) {
    TC_ASSERT(func);
    return func(ds, i, j, k, l);
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
