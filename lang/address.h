#pragma once

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <dlfcn.h>
#include <set>
#include "../headers/common.h"

TC_NAMESPACE_BEGIN
namespace Tlang {

template <typename T>
using Handle = std::shared_ptr<T>;

class Expr;

struct Address {
  int64 buffer_id;
  int64 coeff_i;
  int64 coeff_imax;
  int64 coeff_const;  // offset

  // AOSOA: i / a * b
  int64 coeff_aosoa_group_size;
  int64 coeff_aosoa_stride;

  TC_IO_DEF(buffer_id,
            coeff_i,
            coeff_imax,
            coeff_const,
            coeff_aosoa_stride,
            coeff_aosoa_group_size);

  Address() {
    buffer_id = -1;
    coeff_i = 0;
    coeff_imax = 0;
    coeff_const = 0;
    coeff_aosoa_group_size = 0;
    coeff_aosoa_stride = 0;
  }

  bool initialized() {
    return buffer_id != -1;
  }

  TC_FORCE_INLINE bool same_type(Address o) {
    return buffer_id == o.buffer_id && coeff_i == o.coeff_i &&
           coeff_imax == o.coeff_imax &&
           coeff_aosoa_group_size == o.coeff_aosoa_group_size &&
           coeff_aosoa_stride == o.coeff_aosoa_stride;
  }

  TC_FORCE_INLINE bool operator==(Address o) {
    return buffer_id == o.buffer_id && coeff_i == o.coeff_i &&
           coeff_imax == o.coeff_imax && coeff_const == o.coeff_const &&
           coeff_aosoa_group_size == o.coeff_aosoa_group_size &&
           coeff_aosoa_stride == o.coeff_aosoa_group_size;
  }

  TC_FORCE_INLINE int64 offset() {
    return coeff_const;
  }

  int64 eval(int64 i, int64 n) {
    TC_ASSERT(initialized());
    if (coeff_aosoa_stride != 0) {
      return coeff_i * i + coeff_imax * n + coeff_const +
             (i / coeff_aosoa_group_size) * coeff_aosoa_stride;
    } else {
      return coeff_i * i + coeff_imax * n + coeff_const;
    }
  }
};

struct AddrNode {
  std::vector<Handle<AddrNode>> ch;
  int depth;
  Address *addr;

  int group_size;
  int repeat_factor;
  int num_variables;
  int offset;
  int buffer_id;
  int coeff_i;
  // repeat included
  int data_size;

  AddrNode(int depth, Address *addr = nullptr) : depth(depth), addr(addr) {
    num_variables = 0;
    if (addr) {
      num_variables += 1;
    }
    offset = 0;
    repeat_factor = 1;
  }

  void materialize() {
    TC_ASSERT(bool(addr == nullptr) ^ bool(ch.size() == 0));
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

  void set() {
    int coeff_imax = 0;
    int buffer_id = 0;
    int bundle_num_variables = -1;
    std::function<void(AddrNode *)> walk = [&](AddrNode *node) {
      if (node->addr) {
        node->addr->buffer_id = buffer_id;
        node->addr->coeff_i = node->coeff_i;
        node->addr->coeff_imax = coeff_imax;
        node->addr->coeff_aosoa_group_size = group_size;
        node->addr->coeff_const = node->offset;
        // Note: use root->data_size here
        node->addr->coeff_aosoa_stride =
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

  AddrNode &place(Expr &expr);

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
    fmt::print("  num_variables={} data_size={} repeat={} offset={} addr={}\n",
               num_variables, data_size, repeat_factor, offset, (uint64)addr);
    for (auto c : ch) {
      c->print();
    }
  }
};
}

TC_NAMESPACE_END
