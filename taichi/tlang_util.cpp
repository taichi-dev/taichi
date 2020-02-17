// Definitions of utility functions and enums

#include "tlang_util.h"
#include <taichi/system/timer.h>
#include <taichi/math/linalg.h>

TI_NAMESPACE_BEGIN

namespace Tlang {

CompileConfig default_compile_config;

real get_cpu_frequency() {
  static real cpu_frequency = 0;
  if (cpu_frequency == 0) {
    uint64 cycles = Time::get_cycles();
    Time::sleep(1);
    uint64 elapsed_cycles = Time::get_cycles() - cycles;
    auto frequency = real(std::round(elapsed_cycles / 1e8_f64) / 10.0_f64);
    TI_INFO("CPU frequency = {:.2f} GHz ({} cycles per second)", frequency,
            elapsed_cycles);
    cpu_frequency = frequency;
  }
  return cpu_frequency;
}

real default_measurement_time = 1;

real measure_cpe(std::function<void()> target,
                 int64 elements_per_call,
                 real time_second) {
  if (time_second == 0) {
    target();
    return std::numeric_limits<real>::quiet_NaN();
  }
  // first make rough estimate of run time.
  int64 batch_size = 1;
  while (true) {
    float64 t = Time::get_time();
    for (int64 i = 0; i < batch_size; i++) {
      target();
    }
    t = Time::get_time() - t;
    if (t < 0.05 * time_second) {
      batch_size *= 2;
    } else {
      break;
    }
  }

  int64 total_batches = 0;
  float64 start_t = Time::get_time();
  while (Time::get_time() - start_t < time_second) {
    for (int i = 0; i < batch_size; i++) {
      target();
    }
    total_batches += batch_size;
  }
  auto elasped_cycles =
      (Time::get_time() - start_t) * 1e9_f64 * get_cpu_frequency();
  return elasped_cycles / float64(total_batches * elements_per_call);
}

int default_simd_width(Arch arch) {
  if (arch == Arch::x86_64) {
    return default_simd_width_x86_64;
  } else if (arch == Arch::cuda) {
    return 32;
  } else {
    TI_NOT_IMPLEMENTED;
    return -1;
  }
}

std::string data_type_name(DataType t) {
  static std::map<DataType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_DATA_TYPE(i, j) type_names[DataType::i] = #j;
    REGISTER_DATA_TYPE(f16, float16);
    REGISTER_DATA_TYPE(f32, float32);
    REGISTER_DATA_TYPE(f64, float64);
    REGISTER_DATA_TYPE(i1, int1);
    REGISTER_DATA_TYPE(i8, int8);
    REGISTER_DATA_TYPE(i16, int16);
    REGISTER_DATA_TYPE(i32, int32);
    REGISTER_DATA_TYPE(i64, int64);
    REGISTER_DATA_TYPE(u8, uint8);
    REGISTER_DATA_TYPE(u16, uint16);
    REGISTER_DATA_TYPE(u32, uint32);
    REGISTER_DATA_TYPE(u64, uint64);
    REGISTER_DATA_TYPE(ptr, void_pointer);
    REGISTER_DATA_TYPE(none, none);
    REGISTER_DATA_TYPE(unknown, unknown);
#undef REGISTER_DATA_TYPE
  }
  return type_names[t];
}

int data_type_size(DataType t) {
  static std::map<DataType, int> type_sizes;
  if (type_sizes.empty()) {
#define REGISTER_DATA_TYPE(i, j) type_sizes[DataType::i] = sizeof(j);
    type_sizes[DataType::f16] = 2;
    REGISTER_DATA_TYPE(f32, float32);
    REGISTER_DATA_TYPE(f64, float64);
    REGISTER_DATA_TYPE(i8, bool);
    REGISTER_DATA_TYPE(i8, int8);
    REGISTER_DATA_TYPE(i16, int16);
    REGISTER_DATA_TYPE(i32, int32);
    REGISTER_DATA_TYPE(i64, int64);
    REGISTER_DATA_TYPE(u8, uint8);
    REGISTER_DATA_TYPE(u16, uint16);
    REGISTER_DATA_TYPE(u32, uint32);
    REGISTER_DATA_TYPE(u64, uint64);
    type_sizes[DataType::ptr] = sizeof(void *);
    type_sizes[DataType::none] = 0;
    type_sizes[DataType::unknown] = -1;
#undef REGISTER_DATA_TYPE
  }
  return type_sizes[t];
}

std::string data_type_short_name(DataType t) {
  static std::map<DataType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_DATA_TYPE(i) type_names[DataType::i] = #i;
    REGISTER_DATA_TYPE(f16);
    REGISTER_DATA_TYPE(f32);
    REGISTER_DATA_TYPE(f64);
    REGISTER_DATA_TYPE(i1);
    REGISTER_DATA_TYPE(i8);
    REGISTER_DATA_TYPE(i16);
    REGISTER_DATA_TYPE(i32);
    REGISTER_DATA_TYPE(i64);
    REGISTER_DATA_TYPE(u8);
    REGISTER_DATA_TYPE(u16);
    REGISTER_DATA_TYPE(u32);
    REGISTER_DATA_TYPE(u64);
    REGISTER_DATA_TYPE(ptr);
    REGISTER_DATA_TYPE(none);
    REGISTER_DATA_TYPE(unknown);
#undef REGISTER_DATA_TYPE
  }
  return type_names[t];
}

std::string snode_type_name(SNodeType t) {
  static std::map<SNodeType, std::string> type_names;
  if (type_names.empty()) {
#define PER_SNODE(i) type_names[SNodeType::i] = #i;
#include "inc/snodes.inc.h"
#undef PER_SNODE
  }
  return type_names[t];
}

std::string unary_op_type_name(UnaryOpType type) {
  static std::map<UnaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define PER_UNARY_OP(i) type_names[UnaryOpType::i] = #i;
#include "taichi/inc/unary_op.inc.h"

#undef PER_UNARY_OP
  }
  return type_names[type];
}

std::string binary_op_type_name(BinaryOpType type) {
  static std::map<BinaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define PER_BINARY_OP(x) type_names[BinaryOpType::x] = #x;
#include "inc/binary_op.inc.h"
#undef PER_BINARY_OP
  }
  return type_names[type];
}

std::string binary_op_type_symbol(BinaryOpType type) {
  static std::map<BinaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i, s) type_names[BinaryOpType::i] = #s;
    REGISTER_TYPE(mul, *);
    REGISTER_TYPE(add, +);
    REGISTER_TYPE(sub, -);
    REGISTER_TYPE(div, /);
    REGISTER_TYPE(mod, %);
    REGISTER_TYPE(max, max);
    REGISTER_TYPE(min, min);
    REGISTER_TYPE(atan2, atan2);
    REGISTER_TYPE(cmp_lt, <);
    REGISTER_TYPE(cmp_le, <=);
    REGISTER_TYPE(cmp_gt, >);
    REGISTER_TYPE(cmp_ge, >=);
    REGISTER_TYPE(cmp_ne, !=);
    REGISTER_TYPE(cmp_eq, ==);
    REGISTER_TYPE(bit_and, &);
    REGISTER_TYPE(bit_or, |);
    REGISTER_TYPE(bit_xor, ^);
    REGISTER_TYPE(pow, pow);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string ternary_type_name(TernaryOpType type) {
  static std::map<TernaryOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[TernaryOpType::i] = #i;
    REGISTER_TYPE(select);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string atomic_op_type_name(AtomicOpType type) {
  static std::map<AtomicOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[AtomicOpType::i] = #i;
    REGISTER_TYPE(add);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string snode_op_type_name(SNodeOpType type) {
  static std::map<SNodeOpType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[SNodeOpType::i] = #i;
    REGISTER_TYPE(is_active);
    REGISTER_TYPE(length);
    REGISTER_TYPE(activate);
    REGISTER_TYPE(deactivate);
    REGISTER_TYPE(append);
    REGISTER_TYPE(clear);
    REGISTER_TYPE(undefined);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string CompileConfig::compiler_config() {
  std::string cmd;
#if defined(OPENMP_FOUND)
  std::string omp_flag = "-fopenmp -DTLANG_WITH_OPENMP";
#else
  std::string omp_flag = "";
#endif

#if defined(TI_PLATFORM_OSX)
  std::string linking = "-undefined dynamic_lookup";
#else
  std::string linking = "-ltaichi_core";
#endif

  std::string include_flag;
  std::string link_flag;

  if (is_release()) {
    linking = fmt::format("-L{}/lib -ltaichi_core", get_python_package_dir());
    // TODO: this is useless now since python packages no longer support legacy
    // backends
    include_flag = fmt::format("-I{}/", get_python_package_dir());
  } else {
    linking = fmt::format(" -L{}/build -ltaichi_core ", get_repo_dir());
    include_flag = fmt::format(" -I{}/ ", get_repo_dir());
  }
  if (arch == Arch::x86_64) {
    cmd = fmt::format(
        "{} -std=c++14 -shared -fPIC {} -march=native -mfma {} "
        "-ffp-contract=fast "
        "{} -Wall -g -DTLANG_CPU "
        "-lstdc++  {} {}",
        compiler_name(), gcc_opt_flag(), include_flag, omp_flag, linking,
        extra_flags);
  } else {
    cmd = fmt::format(
        "nvcc -g -lineinfo -std=c++14 -shared {} -Xcompiler \"-fPIC "
        "-march=native \" "
        "--use_fast_math -arch=compute_61 -code=sm_61,compute_61 "
        "--ptxas-options=-allow-expensive-optimizations=true,-O3,-v -I "
        "{}/ -I/usr/local/cuda/include/ -ccbin {} "
        " -lstdc++ {} {} "
        "-DTLANG_GPU {} ",
        gcc_opt_flag(), get_repo_dir(), "g++-6", include_flag, linking,
        extra_flags);
  }
  return cmd;
}

std::string CompileConfig::preprocess_cmd(const std::string &input,
                                          const std::string &output,
                                          const std::string &extra_flags,
                                          bool verbose) {
  std::string cmd = compiler_config();
  std::string io = fmt::format(" {} -E {} -o {} ", extra_flags, input, output);
  if (!verbose) {
    io += " 2> /dev/null ";
  }
  return cmd + io;
}

std::string CompileConfig::compile_cmd(const std::string &input,
                                       const std::string &output,
                                       const std::string &extra_flags,
                                       bool verbose) {
  std::string cmd = compiler_config();
  std::string io = fmt::format(" {} {} -o {} ", extra_flags, input, output);

  cmd += io;

  if (!verbose) {
    cmd += fmt::format(" 2> {}.log", input);
  }
  return cmd;
}

bool command_exist(const std::string &command) {
#if defined(TI_PLATFORM_UNIX)
  if (std::system(fmt::format("which {} > /dev/null 2>&1", command).c_str())) {
    return false;
  } else {
    return true;
  }
#else
  if (std::system(fmt::format("where {} >nul 2>nul", command).c_str())) {
    return false;
  } else {
    return true;
  }
#endif
}

CompileConfig::CompileConfig() {
  arch = Arch::x86_64;
  simd_width = default_simd_width(arch);
  internal_optimization = true;
  external_optimization_level = 3;
  print_ir = false;
  print_accessor_ir = false;
  use_llvm = true;
  print_struct_llvm_ir = false;
  print_kernel_llvm_ir = false;
  print_kernel_llvm_ir_optimized = false;
  demote_dense_struct_fors = true;
  max_vector_width = 8;
  force_vectorized_global_load = false;
  force_vectorized_global_store = false;
  debug = false;
#if defined(TI_PLATFORM_OSX)
  gcc_version = -1;
#else
  gcc_version = -2;  // not 7 for faster compilation
                     // Use clang for faster speed
#endif
  if (!use_llvm) {
    if (gcc_version == -2 && !command_exist("clang-7")) {
      TI_WARN("Command clang-7 not found. Attempting clang");
      gcc_version = -1;
    }
    if (gcc_version == -1 && !command_exist("clang")) {
      TI_WARN("Command clang not found. Attempting gcc-6");
      gcc_version = 6;
    }
  }
  lazy_compilation = true;
  serial_schedule = false;
  simplify_before_lower_access = true;
  lower_access = true;
  simplify_after_lower_access = true;
  attempt_vectorized_load_cpu = true;
  default_fp = DataType::f32;
  default_ip = DataType::i32;
  verbose_kernel_launches = false;
  enable_profiler = false;
  default_cpu_block_dim = 0;  // 0 = adaptive
  default_gpu_block_dim = 64;
  verbose = true;
  fast_math = true;
}

std::string CompileConfig::compiler_name() {
  if (gcc_version == -1) {
    return "clang";
  } else if (gcc_version == -2) {
    return "clang-7";
  } else {
    return fmt::format("gcc-{}", gcc_version);
  }
}

std::string CompileConfig::gcc_opt_flag() {
  TI_ASSERT(0 <= external_optimization_level &&
            external_optimization_level < 5);
  if (external_optimization_level < 4) {
    return fmt::format("-O{}", external_optimization_level);
  } else
    return "-Ofast";
}

DataType promoted_type(DataType a, DataType b) {
  std::map<std::pair<DataType, DataType>, DataType> mapping;
  if (mapping.empty()) {
#define TRY_SECOND(x, y)                                            \
  mapping[std::make_pair(get_data_type<x>(), get_data_type<y>())] = \
      get_data_type<decltype(std::declval<x>() + std::declval<y>())>();
#define TRY_FIRST(x)      \
  TRY_SECOND(x, float32); \
  TRY_SECOND(x, float64); \
  TRY_SECOND(x, int8);    \
  TRY_SECOND(x, int16);   \
  TRY_SECOND(x, int32);   \
  TRY_SECOND(x, int64);   \
  TRY_SECOND(x, uint8);   \
  TRY_SECOND(x, uint16);  \
  TRY_SECOND(x, uint32);  \
  TRY_SECOND(x, uint64);

    TRY_FIRST(float32);
    TRY_FIRST(float64);
    TRY_FIRST(int8);
    TRY_FIRST(int16);
    TRY_FIRST(int32);
    TRY_FIRST(int64);
    TRY_FIRST(uint8);
    TRY_FIRST(uint16);
    TRY_FIRST(uint32);
    TRY_FIRST(uint64);
  }
  return mapping[std::make_pair(a, b)];
}
}  // namespace Tlang

void initialize_benchmark() {
  // CoreState::set_trigger_gdb_when_crash(true);
  Tlang::get_cpu_frequency();
  static bool initialized = false;
  if (initialized) {
    return;
  }
  initialized = true;
#if defined(TI_PLATFORM_LINUX)
  std::ifstream noturbo("/sys/devices/system/cpu/intel_pstate/no_turbo");
  char c;
  noturbo >> c;
  TI_WARN_IF(c != '1',
             "You seem to be running the benchmark with Intel Turboboost.");
#endif
}

TI_NAMESPACE_END
