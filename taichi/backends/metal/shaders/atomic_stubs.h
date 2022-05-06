#pragma once

using atomic_int = int;
using atomic_uint = unsigned int;

template <typename T>
struct _atomic {};

namespace metal {

using memory_order = bool;

// Defined in atomic_stub.cpp
extern memory_order memory_order_relaxed;

}  // namespace metal

template <typename T>
bool atomic_compare_exchange_weak_explicit(T *object,
                                           T *expected,
                                           T desired,
                                           metal::memory_order,
                                           metal::memory_order) {
  const T val = *object;
  if (val == *expected) {
    *object = desired;
    return true;
  }
  *expected = val;
  return false;
}

template <typename T>
bool atomic_exchange_explicit(T *object, T desired, metal::memory_order) {
  const T val = *object;
  *object = desired;
  return val;
}

template <typename T>
bool atomic_fetch_or_explicit(T *object, T operand, metal::memory_order) {
  const T result = *object;
  *object = (result | operand);
  return result;
}

template <typename T>
bool atomic_fetch_and_explicit(T *object, T operand, metal::memory_order) {
  const T result = *object;
  *object = (result & operand);
  return result;
}

template <typename T>
T atomic_fetch_add_explicit(T *object, T operand, metal::memory_order) {
  const T result = *object;
  *object += operand;
  return result;
}

template <typename T>
T atomic_fetch_max_explicit(T *object, T operand, metal::memory_order) {
  const T result = *object;
  *object = (operand > result) ? operand : result;
  return result;
}

template <typename T>
T atomic_fetch_min_explicit(T *object, T operand, metal::memory_order) {
  const T result = *object;
  *object = (operand < result) ? operand : result;
  return result;
}

template <typename T>
T atomic_load_explicit(T *object, metal::memory_order) {
  return *object;
}

template <typename T>
void atomic_store_explicit(T *object, T desired, metal::memory_order) {
  *object = desired;
}
