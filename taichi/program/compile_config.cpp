#include "compile_config.h"

TLANG_NAMESPACE_BEGIN

CompileConfig::CompileConfig() {
  arch = Arch::x64;
  simd_width = default_simd_width(arch);
  external_optimization_level = 3;
  print_ir = false;
  print_accessor_ir = false;
  use_llvm = true;
  print_struct_llvm_ir = false;
  print_kernel_llvm_ir = false;
  print_kernel_llvm_ir_optimized = false;
  demote_dense_struct_fors = true;
  max_vector_width = 8;
  debug = false;
  lazy_compilation = true;
  serial_schedule = false;
  simplify_before_lower_access = true;
  lower_access = true;
  simplify_after_lower_access = true;
  default_fp = DataType::f32;
  default_ip = DataType::i32;
  verbose_kernel_launches = false;
  enable_profiler = false;
  default_cpu_block_dim = 0;  // 0 = adaptive
  default_gpu_block_dim = 64;
  verbose = true;
  fast_math = true;

  device_memory_GB = 0;
  device_memory_fraction = 0.5;  // by default, use 50% GPU memory
}

TLANG_NAMESPACE_END
