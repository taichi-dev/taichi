#include "compile_config.h"

#include <thread>
#include "taichi/rhi/arch.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {

CompileConfig::CompileConfig() {
  arch = host_arch();
  simd_width = default_simd_width(arch);
  opt_level = 1;
  external_optimization_level = 3;
  print_ir = false;
  print_preprocessed_ir = false;
  print_accessor_ir = false;
  use_llvm = true;
  demote_dense_struct_fors = true;
  advanced_optimization = true;
  constant_folding = true;
  max_vector_width = 8;
  debug = false;
  cfg_optimization = true;
  check_out_of_bound = false;
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
  flatten_if = false;
  make_thread_local = true;
  make_block_local = true;
  detect_read_only = true;
  real_matrix_scalarize = true;
  half2_vectorization = false;
  make_cpu_multithreading_loop = true;

  saturating_grid_dim = 0;
  max_block_dim = 0;
  cpu_max_num_threads = std::thread::hardware_concurrency();
  random_seed = 0;

  // LLVM backend options:
  print_struct_llvm_ir = false;
  print_kernel_llvm_ir = false;
  print_kernel_asm = false;
  print_kernel_amdgcn = false;
  print_kernel_llvm_ir_optimized = false;

  // CUDA/AMDGPU backend options:
  device_memory_GB = 1;  // by default, preallocate 1 GB GPU memory
  device_memory_fraction = 0.0;
}

void CompileConfig::fit() {
  if (debug) {
    // TODO: allow users to run in debug mode without out-of-bound checks
    check_out_of_bound = true;
  }
  if (arch_uses_spirv(arch)) {
    demote_dense_struct_fors = true;
  }
  offline_cache::disable_offline_cache_if_needed(this);
}

}  // namespace taichi::lang
