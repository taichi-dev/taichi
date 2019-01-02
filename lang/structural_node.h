#pragma once
#include "expr.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

TC_FORCE_INLINE int32 constexpr operator"" _bits(unsigned long long a) {
  return 1 << a;
}

struct IndexExtractor {
  int start, num_bits;

  IndexExtractor() {
    start = 0;
    num_bits = 0;
  }
};

// "Structural" nodes
struct SNode {
  std::vector<Handle<SNode>> ch;

  IndexExtractor extractors[max_num_indices];
  int taken_bits[max_num_indices];  // counting from the tail

  static int counter;
  int id;
  int depth;

  Expr addr;
  SNode *parent;

  using AccessorFunction = void *(*)(void *, int);
  AccessorFunction func;

  int group_size;
  int repeat_factor;
  int num_variables;
  int offset;
  int buffer_id;
  int coeff_i;
  std::string node_type_name;
  int64 n;

  // repeat included
  int data_size;

  SNodeType type;

  SNode() {
    id = counter++;
  }

  SNode(int depth, SNodeType t) : depth(depth), type(t) {
    id = counter++;
    std::memset(taken_bits, 0, sizeof(taken_bits));
    func = nullptr;
    parent = nullptr;
  }

  SNode &insert_children(SNodeType t) {
    if (this->type != SNodeType::forked) {
      TC_ASSERT(ch.size() == 0);
    }
    ch.push_back(create(depth + 1, t));
    ch.back()->parent = this;
    return *ch.back();
  }

  // Let us deal with 1D case first
  // SNodes maintains how flattened index bits are taken from indices
  SNode &fixed(std::vector<Expr> indices, std::vector<int> size) {
    TC_ASSERT(size.size() == 1);
    TC_ASSERT(bit::is_power_of_two(size[0]));
    auto &new_node = insert_children(SNodeType::fixed);
    new_node.n = size[0];
    for (int i = 0; i < (int)indices.size(); i++) {
      auto &ind = indices[i];
      TC_ASSERT(ind->lanes == 1);
      new_node.extractors[ind->index_id(0)].num_bits = size[i];
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

  template <typename... Args>
  SNode &place(Expr &expr, Args &&... args) {
    return place(expr).place(std::forward<Args>(args)...);
  }

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
    TC_ASSERT(expr);
    auto &child = insert_children(SNodeType::place);
    expr->new_addresses(0) = &child;
    child.addr.set(expr);
    return *this;
  }

  TC_FORCE_INLINE void *evaluate(void *ds, int i) {
    TC_ASSERT(func);
    return func(ds, i);
  }
};

TLANG_NAMESPACE_END
