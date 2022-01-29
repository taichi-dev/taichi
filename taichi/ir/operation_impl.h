#pragma once

#include "taichi/ir/operation.h"

namespace taichi {
namespace lang {

class StaticTraits {
  StaticTraits() = default;
  void init();
  static inline std::unique_ptr<StaticTraits> instance_;

 public:
  Trait *primitive;
  Trait *custom;
  Trait *scalar;
  Trait *real;
  Trait *integral;

  static const StaticTraits &get();
};

class InternalOps {
  InternalOps() = default;
  void init();
  static inline std::unique_ptr<InternalOps> instance_;

 public:
  Operation *thread_index;
  Operation *insert_triplet;
  Operation *do_nothing;
  Operation *refresh_counter;

  // binops

  // arith - i/f
  Operation *add;
  Operation *sub;
  Operation *mul;
  Operation *div;
  Operation *truediv;
  Operation *floordiv;
  Operation *pow;
  Operation *max;
  Operation *min;

  // bitwise - i
  Operation *bit_and;
  Operation *bit_or;
  Operation *bit_xor;
  Operation *bit_shl;
  Operation *bit_shr;
  Operation *bit_sar;

  // compare - i/f, ret i32
  Operation *cmp_lt;
  Operation *cmp_le;
  Operation *cmp_gt;
  Operation *cmp_ge;
  Operation *cmp_eq;
  Operation *cmp_ne;

  // other binops
  Operation *atan2;  // f
  Operation *mod;    // i

  // unops

  // f
  Operation *floor;
  Operation *ceil;
  Operation *round;
  Operation *sin;
  Operation *asin;
  Operation *cos;
  Operation *acos;
  Operation *tan;
  Operation *tanh;
  Operation *exp;
  Operation *log;
  Operation *sqrt;
  Operation *rsqrt;
  Operation *sgn;

  // i/f
  Operation *neg;
  Operation *abs;

  // i
  Operation *bit_not;
  Operation *logic_not;

  // atomics

  // i/f
  Operation *atomic_add;
  Operation *atomic_sub;
  Operation *atomic_max;
  Operation *atomic_min;

  // i
  Operation *atomic_bit_and;
  Operation *atomic_bit_or;
  Operation *atomic_bit_xor;

  // ternary

  Operation *select;

  static const InternalOps &get();
};

class TestInternalOps {
  TestInternalOps() = default;
  void init();
  static inline std::unique_ptr<TestInternalOps> instance_;

 public:
  Operation *test_stack;
  Operation *test_active_mask;
  Operation *test_shfl;
  Operation *test_list_manager;
  Operation *test_node_allocator;
  Operation *test_node_allocator_gc_cpu;
  Operation *test_internal_func_args;

  static const TestInternalOps &get();
};

}  // namespace lang
}  // namespace taichi
