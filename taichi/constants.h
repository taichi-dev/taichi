#pragma once

#include <cstddef>

constexpr int taichi_max_num_indices = 8;
constexpr int taichi_max_num_args = 8;
constexpr int taichi_max_num_snodes = 1024;
constexpr std::size_t taichi_global_tmp_buffer_size = 1024 * 1024;
constexpr int taichi_max_num_mem_requests = 1024 * 64;
constexpr std::size_t taichi_page_size = 4096;
