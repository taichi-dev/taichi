#include "compile_config.h"

#include <thread>

TLANG_NAMESPACE_BEGIN

CompileConfig::CompileConfig() {
  arch = host_arch();
  simd_width = default_simd_width(arch);
  opt_level = 1;
  external_optimization_level = 3;
  packed = false;
  print_ir = false;
  print_preprocessed_ir = false;
  print_accessor_ir = false;
  print_evaluator_ir = false;
  print_benchmark_stat = false;
  use_llvm = true;
  demote_dense_struct_fors = true;
  advanced_optimization = true;
  constant_folding = true;
  max_vector_width = 8;
  debug = false;
  cfg_optimization = true;
  check_out_of_bound = false;
  validate_autodiff = false;
  lazy_compilation = true;
  serial_schedule = false;
  simplify_before_lower_access = true;
  lower_access = true;
  simplify_after_lower_access = true;
  move_loop_invariant_outside_if = false;
  default_fp = PrimitiveType::f32;
  default_ip = PrimitiveType::i32;
  default_up = PrimitiveType::u32;
  verbose_kernel_launches = false;
  kernel_profiler = false;
  default_cpu_block_dim = 32;
  cpu_block_dim_adaptive = true;
  default_gpu_block_dim = 128;
  gpu_max_reg = 0;  // 0 means using the default value from the CUDA driver.
  verbose = true;
  fast_math = true;
  dynamic_index = false;
  flatten_if = false;
  make_thread_local = true;
  make_block_local = true;
  detect_read_only = true;
  ndarray_use_cached_allocator = true;
  use_mesh = false;
  real_matrix = false;

  saturating_grid_dim = 0;
  max_block_dim = 0;
  cpu_max_num_threads = std::thread::hardware_concurrency();
  random_seed = 0;

  // LLVM backend options:
  print_struct_llvm_ir = false;
  print_kernel_llvm_ir = false;
  print_kernel_nvptx = false;
  print_kernel_llvm_ir_optimized = false;

  // CUDA backend options:
  device_memory_GB = 1;  // by default, preallocate 1 GB GPU memory
  device_memory_fraction = 0.0;

  // C backend options:
  cc_compile_cmd = "gcc -Wc99-c11-compat -c -o '{}' '{}' -O3";
  cc_link_cmd = "gcc -shared -fPIC -o '{}' '{}'";
}

TLANG_NAMESPACE_END
