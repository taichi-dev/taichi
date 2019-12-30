#pragma once

constexpr int taichi_max_num_indices = 8;
constexpr int taichi_max_num_args = 8;
constexpr int taichi_max_num_snodes = 1024;
constexpr int taichi_max_num_global_vars = 1024 * 1024;

using assert_failed_type = void (*)(const char *);
