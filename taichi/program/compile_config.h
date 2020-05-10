#pragma once

#include "taichi/lang_util.h"
#include "arch.h"

TLANG_NAMESPACE_BEGIN

// TODO(xumingkuan): Temporary variable for benchmarking.
// TODO(xumingkuan): Will be removed in the future.
extern bool advanced_optimization;

struct CompileConfig {
  Arch arch;
  bool debug;
  bool check_out_of_bound;
  int simd_width;
  bool lazy_compilation;
  int external_optimization_level;
  int max_vector_width;
  bool print_ir;
  bool print_accessor_ir;
  bool print_benchmark_stat;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool demote_dense_struct_fors;
  bool use_llvm;
  bool print_struct_llvm_ir;
  bool print_kernel_llvm_ir;
  bool print_kernel_llvm_ir_optimized;
  bool verbose_kernel_launches;
  bool enable_profiler;
  bool verbose;
  bool fast_math;
  bool use_unified_memory;
  bool async;
  DataType default_fp;
  DataType default_ip;
  std::string extra_flags;
  int default_cpu_block_dim;
  int default_gpu_block_dim;

  float64 device_memory_GB;
  float64 device_memory_fraction;

  CompileConfig();
};

extern CompileConfig default_compile_config;

TLANG_NAMESPACE_END
