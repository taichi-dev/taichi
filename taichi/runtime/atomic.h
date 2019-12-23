#pragma once

int32 atomic_exchange_i32(volatile int32 *dest, int32 val) {
  i32 ret;
  __atomic_exchange(dest, &val, &ret, std::memory_order::memory_order_seq_cst);
  return ret;
}

int32 atomic_add_i32(volatile int32 *dest, int32 val) {
  return __atomic_fetch_add(dest, val, std::memory_order::memory_order_seq_cst);
}

int64 atomic_add_i64(volatile int64 *dest, int64 val) {
  return __atomic_fetch_add(dest, val, std::memory_order::memory_order_seq_cst);
}

uint64 atomic_add_u64(volatile uint64 *dest, uint64 val) {
  return __atomic_fetch_add(dest, val, std::memory_order::memory_order_seq_cst);
}

uint64 atomic_or_u64(volatile uint64 *dest, uint64 val) {
  return __atomic_fetch_or(dest, val, std::memory_order::memory_order_seq_cst);
}

uint64 atomic_and_u64(volatile uint64 *dest, uint64 val) {
  return __atomic_fetch_and(dest, val, std::memory_order::memory_order_seq_cst);
}

float32 atomic_add_cpu_f32(volatile float32 *dest, float32 inc) {
  float32 old_val;
  float32 new_val;
  do {
    old_val = *dest;
    new_val = old_val + inc;
  } while (!__atomic_compare_exchange(dest, &old_val, &new_val, true,
                                      std::memory_order::memory_order_seq_cst,
                                      std::memory_order::memory_order_seq_cst));
  return old_val;
}

float64 atomic_add_cpu_f64(volatile float64 *dest, float64 inc) {
  float64 old_val;
  float64 new_val;
  do {
    old_val = *dest;
    new_val = old_val + inc;
  } while (!__atomic_compare_exchange(dest, &old_val, &new_val, true,
                                      std::memory_order::memory_order_seq_cst,
                                      std::memory_order::memory_order_seq_cst));
  return old_val;
}
