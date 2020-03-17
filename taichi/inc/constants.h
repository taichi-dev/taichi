#pragma once

#include <cstddef>

constexpr int taichi_max_num_indices = 8;
constexpr int taichi_max_num_args = 8;
constexpr int taichi_max_num_snodes = 1024;
constexpr int taichi_max_gpu_block_dim = 1024;
constexpr std::size_t taichi_global_tmp_buffer_size = 1024 * 1024;
constexpr int taichi_max_num_mem_requests = 1024 * 64;
constexpr std::size_t taichi_page_size = 4096;
constexpr std::size_t taichi_max_message_length = 2048;

template <typename T, typename G>
T taichi_union_cast_with_different_sizes(G g) {
  union {
    T t;
    G g;
  } u;
  u.g = g;
  return u.t;
}

template <typename T, typename G>
T taichi_union_cast(G g) {
  static_assert(sizeof(T) == sizeof(G));
  return taichi_union_cast_with_different_sizes<T>(g);
}
