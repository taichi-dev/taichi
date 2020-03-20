#pragma once

// These instructions should be replaced by CUDA intrinsics on GPUs

#define DEFINE_ATOMIC_EXCHANGE(T)                               \
  T atomic_exchange_##T(volatile T *dest, T val) {              \
    T ret;                                                      \
    __atomic_exchange(dest, &val, &ret,                         \
                      std::memory_order::memory_order_seq_cst); \
    return ret;                                                 \
  }

DEFINE_ATOMIC_EXCHANGE(i32)
DEFINE_ATOMIC_EXCHANGE(i64)
DEFINE_ATOMIC_EXCHANGE(u32)
DEFINE_ATOMIC_EXCHANGE(u64)

#define DEFINE_ATOMIC_OP_INTRINSIC(OP, T)                                \
  T atomic_##OP##_##T(volatile T *dest, T val) {                         \
    return __atomic_fetch_##OP(dest, val,                                \
                               std::memory_order::memory_order_seq_cst); \
  }

DEFINE_ATOMIC_OP_INTRINSIC(add, i32)
DEFINE_ATOMIC_OP_INTRINSIC(add, i64)
DEFINE_ATOMIC_OP_INTRINSIC(and, i32)
DEFINE_ATOMIC_OP_INTRINSIC(and, i64)
DEFINE_ATOMIC_OP_INTRINSIC(and, u32)
DEFINE_ATOMIC_OP_INTRINSIC(and, u64)
DEFINE_ATOMIC_OP_INTRINSIC(or, i32)
DEFINE_ATOMIC_OP_INTRINSIC(or, i64)
DEFINE_ATOMIC_OP_INTRINSIC(or, u32)
DEFINE_ATOMIC_OP_INTRINSIC(or, u64)
DEFINE_ATOMIC_OP_INTRINSIC (xor, i32)
DEFINE_ATOMIC_OP_INTRINSIC (xor, i64)
DEFINE_ATOMIC_OP_INTRINSIC (xor, u32)
DEFINE_ATOMIC_OP_INTRINSIC (xor, u64)

inline f32 add_f32(f32 a, f32 b) {
  return a + b;
}

inline f64 add_f64(f64 a, f64 b) {
  return a + b;
}

inline f32 min_f32(f32 a, f32 b) {
  return b > a ? a : b;
}

inline f64 min_f64(f64 a, f64 b) {
  return b > a ? a : b;
}

inline f32 max_f32(f32 a, f32 b) {
  return b < a ? a : b;
}

inline f64 max_f64(f64 a, f64 b) {
  return b < a ? a : b;
}

#define DEFINE_ATOMIC_OP_COMP_EXCH(OP, T)                                     \
  T atomic_##OP##_##T(volatile T *dest, T inc) {                              \
    T old_val;                                                                \
    T new_val;                                                                \
    do {                                                                      \
      old_val = *dest;                                                        \
      new_val = OP##_##T(old_val, inc);                                       \
    } while (                                                                 \
        !__atomic_compare_exchange(dest, &old_val, &new_val, true,            \
                                   std::memory_order::memory_order_seq_cst,   \
                                   std::memory_order::memory_order_seq_cst)); \
    return old_val;                                                           \
  }

DEFINE_ATOMIC_OP_COMP_EXCH(add, f32)
DEFINE_ATOMIC_OP_COMP_EXCH(add, f64)
DEFINE_ATOMIC_OP_COMP_EXCH(min, f32)
DEFINE_ATOMIC_OP_COMP_EXCH(min, f64)
DEFINE_ATOMIC_OP_COMP_EXCH(min, i32)
DEFINE_ATOMIC_OP_COMP_EXCH(min, i64)
DEFINE_ATOMIC_OP_COMP_EXCH(max, f32)
DEFINE_ATOMIC_OP_COMP_EXCH(max, f64)
DEFINE_ATOMIC_OP_COMP_EXCH(max, i32)
DEFINE_ATOMIC_OP_COMP_EXCH(max, i64)
