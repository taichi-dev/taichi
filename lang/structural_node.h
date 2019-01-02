#pragma once
#include "expr.h"
#include <taichi/common/bit.h>

TLANG_NAMESPACE_BEGIN

TC_FORCE_INLINE int32 constexpr operator"" _bits(unsigned long long a) {
  return 1 << a;
}

// "Structural" nodes
struct SNode {
  std::vector<Handle<SNode>> ch;

  static int counter;
  int id;
  int depth;

  Expr addr;

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
    func = nullptr;
  }

  SNode(int depth, const Expr &addr = Expr()) : depth(depth), addr(addr) {
    func = nullptr;
    id = counter++;
    n = -1;
    num_variables = 0;
    if (addr) {
      num_variables += 1;
    }
    offset = 0;
    repeat_factor = 1;
  }

  void materialize() {
    TC_ASSERT(bool(addr == nullptr) || bool(ch.size() == 0));
    TC_ASSERT(!(bool(addr == nullptr) && bool(ch.size() == 0)));
    if (depth == 1) {
      TC_ASSERT_INFO(n != -1, "Please set n for buffer.");
    }
    if (depth == 2) {  // stream, reset offset
      offset = 0;
    }
    int acc_offset = offset;
    for (auto &c : ch) {
      if (n != -1 && c->n == -1)
        c->n = n;
      c->offset = acc_offset;
      c->materialize();
      num_variables += c->num_variables;
      acc_offset += c->data_size;
    }
    data_size = num_variables * repeat_factor;
    group_size = (ch.size() ? ch[0]->group_size : 1) * repeat_factor;
  }

  void set() {
    int coeff_imax = 0;
    int buffer_id = 0;
    int bundle_num_variables = -1;
    std::function<void(SNode *)> walk = [&](SNode *node) {
      if (node->addr) {
        auto &ad = node->addr->get_address_();  // TODO: remove this hack
        ad.buffer_id = buffer_id;
        ad.n = node->n;
        ad.coeff_i = node->coeff_i;
        ad.coeff_imax = coeff_imax;
        ad.coeff_aosoa_group_size = group_size;
        ad.coeff_const = node->offset;
        // Note: use root->data_size here
        ad.coeff_aosoa_stride =
            group_size * (bundle_num_variables - node->coeff_i);
      }
      for (auto c : node->ch) {
        if (c->depth == 2) {  // stream
          bundle_num_variables = c->num_variables;
        }
        if (c->depth == 1) {  // buffer
          buffer_id = c->buffer_id;
        }
        c->coeff_i = node->num_variables;
        walk(c.get());
        if (c->depth == 1) {  // buffer
          coeff_imax = 0;
        } else if (c->depth == 2) {        // stream
          coeff_imax += c->num_variables;  // stream attr update
        }
      }
    };

    walk(this);
  }

  SNode &group(int id = -1) {
    TC_ASSERT(depth >= 2);
    if (id == -1) {
      auto n = create(depth + 1);
      ch.push_back(n);
      return *n;
    } else {
      while ((int)ch.size() <= id) {
        auto n = create(depth + 1);
        ch.push_back(n);
      }
      return *ch[id];
    }
  }

  SNode &stream(int id = -1) {
    TC_ASSERT(depth == 1);
    if (id == -1) {
      auto n = create(depth + 1);
      ch.push_back(n);
      return *n;
    } else {
      while ((int)ch.size() <= id) {
        auto n = create(depth + 1);
        ch.push_back(n);
      }
      return *ch[id];
    }
  }

  SNode &insert_children(SNodeType t) {
    if (this->type != SNodeType::forked) {
      TC_ASSERT(ch.size() == 0);
    }
    ch.push_back(create(depth + 1, t));
    return *ch.back();
  }

  // Let us deal with 1D case first
  // SNodes maintains how flattened index bits are taken from indices
  SNode &fixed(Expr ind, int size) {
    TC_ASSERT(bit::is_power_of_two(size));
    auto &new_node = insert_children(SNodeType::fixed);
    new_node.n = size;
    return new_node;
  }

  SNode &forked() {
    auto &new_node = insert_children(SNodeType::forked);
    return new_node;
  }

  SNode &repeat(int repeat_factor) {
    this->repeat_factor = repeat_factor;
    return *this;
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

  SNode &range(int64 n) {
    TC_ASSERT(this->depth == 1);
    this->n = n;
    return *this;
  }

  TC_FORCE_INLINE void *evaluate(void *ds, int i) {
    TC_ASSERT(func);
    return func(ds, i);
  }
};

TLANG_NAMESPACE_END
