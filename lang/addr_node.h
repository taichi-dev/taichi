#include "expr.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

struct AddrNode {
  std::vector<Handle<AddrNode>> ch;
  int depth;
  Expr addr;

  int group_size;
  int repeat_factor;
  int num_variables;
  int offset;
  int buffer_id;
  int coeff_i;
  // repeat included
  int data_size;

  void materialize() {
    TC_ASSERT(bool(addr == nullptr) || bool(ch.size() == 0));
    TC_ASSERT(!(bool(addr == nullptr) && bool(ch.size() == 0)));
    if (depth == 2) {  // stream, reset offset
      offset = 0;
    }
    int acc_offset = offset;
    for (auto &c : ch) {
      c->offset = acc_offset;
      c->materialize();
      num_variables += c->num_variables;
      acc_offset += c->data_size;
    }
    data_size = num_variables * repeat_factor;
    group_size = (ch.size() ? ch[0]->group_size : 1) * repeat_factor;
  }

  AddrNode(int depth, const Expr &addr = Expr()) : depth(depth), addr(addr) {
    num_variables = 0;
    if (addr) {
      num_variables += 1;
    }
    offset = 0;
    repeat_factor = 1;
  }

  void set() {
    int coeff_imax = 0;
    int buffer_id = 0;
    int bundle_num_variables = -1;
    std::function<void(AddrNode *)> walk = [&](AddrNode *node) {
      if (node->addr) {
        auto &ad = node->addr->get_address();
        ad.buffer_id = buffer_id;
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

  AddrNode &group(int id = -1) {
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

  AddrNode &stream(int id = -1) {
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

  AddrNode &repeat(int repeat_factor) {
    this->repeat_factor = repeat_factor;
    return *this;
  }

  template <typename... Args>
  AddrNode &place(Expr &expr, Args &&... args) {
    return place(expr).place(std::forward<Args>(args)...);
  }

  template <typename... Args>
  static Handle<AddrNode> create(Args &&... args) {
    return std::make_shared<AddrNode>(std::forward<Args>(args)...);
  }

  void print() {
    for (int i = 0; i < depth; i++) {
      fmt::print("  ");
    }
    fmt::print("num_variables={} data_size={} repeat={} offset={} addr={}\n",
               num_variables, data_size, repeat_factor, offset, (uint64)addr);
    for (auto c : ch) {
      c->print();
    }
  }

  AddrNode &place(Expr &expr) {
    if (!expr) {
      expr = placeholder();
    }
    TC_ASSERT(depth >= 3);
    TC_ASSERT(this->addr == nullptr);
    ch.push_back(create(depth + 1, expr));
    return *this;
  }
};
}
TC_NAMESPACE_END
