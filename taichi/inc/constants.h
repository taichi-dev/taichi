#pragma once

#include <cstddef>

constexpr int taichi_max_num_indices = 8;
// legacy: only used in cc and opengl backends
constexpr int taichi_max_num_args = 8;
// used in llvm backend: only the first 32 arguments can be types.ndarray
// TODO: refine argument passing
constexpr int taichi_max_num_args_total = 64;
constexpr int taichi_max_num_args_extra = 32;
constexpr int taichi_max_num_snodes = 1024;
constexpr int kMaxNumSnodeTreesLlvm = 512;
constexpr int taichi_max_gpu_block_dim = 1024;
constexpr std::size_t taichi_global_tmp_buffer_size = 1024 * 1024;
constexpr int taichi_max_num_mem_requests = 1024 * 64;
constexpr std::size_t taichi_page_size = 4096;
constexpr std::size_t taichi_error_message_max_length = 2048;
constexpr std::size_t taichi_error_message_max_num_arguments = 32;
constexpr std::size_t taichi_result_buffer_entries = 32;
constexpr std::size_t taichi_max_num_ret_value = 30;
// slot for kernel return value
constexpr std::size_t taichi_result_buffer_ret_value_id = 0;
// slot for error code and error message char *
constexpr std::size_t taichi_result_buffer_error_id = 30;
constexpr std::size_t taichi_result_buffer_runtime_query_id = 31;

constexpr int taichi_listgen_max_element_size = 1024;

// use for auto mesh_local to determine shared-mem size per block (in bytes)
// TODO: get this at runtime
constexpr std::size_t default_shared_mem_size = 65536;

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

enum class ExternalArrayLayout { kAOS, kSOA, kNull };

enum class AutodiffMode { kForward, kReverse, kNone, kCheckAutodiffValid };

enum class SNodeGradType { kPrimal, kAdjoint, kDual, kAdjointCheckbit };
