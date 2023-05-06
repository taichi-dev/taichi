#pragma once

#include "taichi/rhi/arch.h"
#include "taichi/util/lang_util.h"

namespace taichi::lang {

struct CompileConfig {
  Arch arch;
  bool debug;
  bool cfg_optimization;
  bool check_out_of_bound;
  bool validate_autodiff;
  int simd_width;
  int opt_level;
  int external_optimization_level;
  int max_vector_width;
  bool print_preprocessed_ir;
  bool print_ir;
  bool print_accessor_ir;
  bool serial_schedule;
  bool simplify_before_lower_access;
  bool lower_access;
  bool simplify_after_lower_access;
  bool move_loop_invariant_outside_if;
  bool cache_loop_invariant_global_vars{true};
  bool demote_dense_struct_fors;
  bool advanced_optimization;
  bool constant_folding;
  bool use_llvm;
  bool verbose_kernel_launches;
  bool kernel_profiler;
  bool timeline{false};
  bool verbose;
  bool fast_math;
  bool flatten_if;
  bool make_thread_local;
  bool make_block_local;
  bool detect_read_only;
  bool real_matrix_scalarize;
  bool half2_vectorization;
  bool make_cpu_multithreading_loop;
  DataType default_fp;
  DataType default_ip;
  DataType default_up;
  std::string extra_flags;
  int default_cpu_block_dim;
  bool cpu_block_dim_adaptive;
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
  bool print_kernel_asm;
  bool print_kernel_amdgcn;

  // CUDA/AMDGPU backend options:
  float64 device_memory_GB;
  float64 device_memory_fraction;

  // Opengl backend options:
  bool allow_nv_shader_extension{true};

  bool quant_opt_store_fusion{true};
  bool quant_opt_atomic_demotion{true};

  // Mesh related.
  // MeshTaichi options
  bool make_mesh_block_local{true};
  bool optimize_mesh_reordered_mapping{true};
  bool mesh_localize_to_end_mapping{true};
  bool mesh_localize_from_end_mapping{false};
  bool mesh_localize_all_attr_mappings{false};
  bool demote_no_access_mesh_fors{true};
  bool experimental_auto_mesh_local{false};
  int auto_mesh_local_default_occupacy{4};

  // Offline cache options
  bool offline_cache{false};
  std::string offline_cache_file_path{get_repo_dir() + "ticache"};
  std::string offline_cache_cleaning_policy{
      "lru"};  // "never"|"version"|"lru"|"fifo"
  int offline_cache_max_size_of_files{100 * 1024 *
                                      1024};   // bytes, default: 100MB
  double offline_cache_cleaning_factor{0.25};  // [0.f, 1.f]

  int num_compile_threads{4};
  std::string vk_api_version;

  size_t cuda_stack_limit{0};

  CompileConfig();

  void fit();
};

extern TI_DLL_EXPORT CompileConfig default_compile_config;

}  // namespace taichi::lang
