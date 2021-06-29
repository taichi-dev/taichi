#pragma once

#include "arch.h"
#include "taichi/lang_util.h"

TLANG_NAMESPACE_BEGIN

struct CompileConfig {
  Arch arch;
  bool debug;
  bool cfg_optimization;
  bool check_out_of_bound;
  int simd_width;
  bool lazy_compilation;
  int external_optimization_level;
  int max_vector_width;
  bool print_ir;
  bool print_accessor_ir;
  bool print_evaluator_ir;
  bool print_benchmark_stat;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool move_loop_invariant_outside_if;
  bool demote_dense_struct_fors;
  bool advanced_optimization;
  bool use_llvm;
  bool verbose_kernel_launches;
  bool kernel_profiler;
  bool timeline{false};
  bool verbose;
  bool fast_math;
  bool async_mode;
  bool flatten_if;
  bool make_thread_local;
  bool make_block_local;
  bool detect_read_only;
  DataType default_fp;
  DataType default_ip;
  std::string extra_flags;
  int default_cpu_block_dim;
  int default_gpu_block_dim;
  int gpu_max_reg;
  int ad_stack_size{0};  // 0 = adaptive
  // The default size when the Taichi compiler is unable to automatically
  // determine the autodiff stack size.
  int default_ad_stack_size{32};

  int saturating_grid_dim;
  int max_block_dim;
  int cpu_max_num_threads;
  int random_seed;

  // LLVM backend options:
  bool print_struct_llvm_ir;
  bool print_kernel_llvm_ir;
  bool print_kernel_llvm_ir_optimized;
  bool print_kernel_nvptx;

  // CUDA backend options:
  bool use_unified_memory;
  float64 device_memory_GB;
  float64 device_memory_fraction;

  // C backend options:
  std::string cc_compile_cmd;
  std::string cc_link_cmd;

  // Async options
  int async_opt_passes{3};
  bool async_opt_fusion{true};
  int async_opt_fusion_max_iter{0};  // 0 means unlimited
  bool async_opt_listgen{true};
  bool async_opt_activation_demotion{true};
  bool async_opt_dse{true};
  bool async_listgen_fast_filtering{true};
  std::string async_opt_intermediate_file;
  // Setting 0 effectively means do not automatically flush
  int async_flush_every{50};
  // Setting 0 effectively means unlimited
  int async_max_fuse_per_task{1};

  bool quant_opt_store_fusion{true};
  bool quant_opt_atomic_demotion{true};

  CompileConfig();
};

extern CompileConfig default_compile_config;

TLANG_NAMESPACE_END
