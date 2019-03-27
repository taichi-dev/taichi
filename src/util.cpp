#include "util.h"
#include <taichi/system/timer.h>
#include <Eigen/Eigen>

TC_NAMESPACE_BEGIN

namespace Tlang {
real get_cpu_frequency() {
  static real cpu_frequency = 0;
  if (cpu_frequency == 0) {
    uint64 cycles = Time::get_cycles();
    Time::sleep(1);
    uint64 elapsed_cycles = Time::get_cycles() - cycles;
    auto frequency = real(std::round(elapsed_cycles / 1e8_f64) / 10.0_f64);
    TC_INFO("CPU frequency = {:.2f} GHz ({} cycles per second)", frequency,
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

std::string get_project_fn() {
  return fmt::format("{}/projects/taichi_lang/", get_repo_dir());
}

int default_simd_width(Arch arch) {
  if (arch == Arch::x86_64) {
    return default_simd_width_x86_64;
  } else if (arch == Arch::gpu) {
    return 32;
  } else {
    TC_NOT_IMPLEMENTED;
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
    REGISTER_DATA_TYPE(i8, int8);
    REGISTER_DATA_TYPE(i16, int16);
    REGISTER_DATA_TYPE(i32, int32);
    REGISTER_DATA_TYPE(i64, int64);
    REGISTER_DATA_TYPE(u8, uint8);
    REGISTER_DATA_TYPE(u16, uint16);
    REGISTER_DATA_TYPE(u32, uint32);
    REGISTER_DATA_TYPE(u64, uint64);
    REGISTER_DATA_TYPE(ptr, pointer);
    REGISTER_DATA_TYPE(none, none);
    REGISTER_DATA_TYPE(unknown, unknown);
#undef REGISTER_DATA_TYPE
  }
  return type_names[t];
}

std::string data_type_short_name(DataType t) {
  static std::map<DataType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_DATA_TYPE(i) type_names[DataType::i] = #i;
    REGISTER_DATA_TYPE(f16);
    REGISTER_DATA_TYPE(f32);
    REGISTER_DATA_TYPE(f64);
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
#define REGISTER_TYPE(i) type_names[SNodeType::i] = #i;
    REGISTER_TYPE(undefined);
    REGISTER_TYPE(root);
    REGISTER_TYPE(fixed);
    REGISTER_TYPE(dynamic);
    REGISTER_TYPE(place);
    REGISTER_TYPE(hashed);
    REGISTER_TYPE(pointer);
    REGISTER_TYPE(indirect);
#undef REGISTER_TYPE
  }
  return type_names[t];
}

std::string unary_type_name(UnaryType type) {
  static std::map<UnaryType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[UnaryType::i] = #i;
    REGISTER_TYPE(neg);
    REGISTER_TYPE(sqrt);
    REGISTER_TYPE(floor);
    REGISTER_TYPE(cast);
    REGISTER_TYPE(abs);
    REGISTER_TYPE(sin);
    REGISTER_TYPE(cos);
    REGISTER_TYPE(inv);
    REGISTER_TYPE(bit_not);
    REGISTER_TYPE(undefined);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string binary_type_name(BinaryType type) {
  static std::map<BinaryType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[BinaryType::i] = #i;
    REGISTER_TYPE(mul);
    REGISTER_TYPE(add);
    REGISTER_TYPE(sub);
    REGISTER_TYPE(div);
    REGISTER_TYPE(mod);
    REGISTER_TYPE(max);
    REGISTER_TYPE(min);
    REGISTER_TYPE(bit_and);
    REGISTER_TYPE(bit_or);
    REGISTER_TYPE(cmp_lt);
    REGISTER_TYPE(cmp_le);
    REGISTER_TYPE(cmp_gt);
    REGISTER_TYPE(cmp_ge);
    REGISTER_TYPE(cmp_ne);
    REGISTER_TYPE(cmp_eq);
#undef REGISTER_TYPE
  }
  return type_names[type];
}


std::string trinary_type_name(TrinaryType type) {
  static std::map<TrinaryType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i) type_names[TrinaryType::i] = #i;
    REGISTER_TYPE(select);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string binary_type_symbol(BinaryType type) {
  static std::map<BinaryType, std::string> type_names;
  if (type_names.empty()) {
#define REGISTER_TYPE(i, s) type_names[BinaryType::i] = #s;
    REGISTER_TYPE(mul, *);
    REGISTER_TYPE(add, +);
    REGISTER_TYPE(sub, -);
    REGISTER_TYPE(div, /);
    REGISTER_TYPE(mod, %);
    REGISTER_TYPE(max, max);
    REGISTER_TYPE(min, min);
    REGISTER_TYPE(cmp_lt, <);
    REGISTER_TYPE(cmp_le, <=);
    REGISTER_TYPE(cmp_gt, >);
    REGISTER_TYPE(cmp_ge, >=);
    REGISTER_TYPE(cmp_ne, !=);
    REGISTER_TYPE(cmp_eq, ==);
    REGISTER_TYPE(bit_and, &&);
    REGISTER_TYPE(bit_or, ||);
#undef REGISTER_TYPE
  }
  return type_names[type];
}

std::string CompileConfig::compile_cmd(const std::string &input,
                                       const std::string &output,
                                       bool verbose) {
  std::string cmd;
  if (arch == Arch::x86_64) {
    cmd = fmt::format(
        "{} {} -std=c++14 -shared -fPIC {} -march=native -mfma -I {}/include "
        "-ffp-contract=fast "
        "-fopenmp -Wall -g -D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_CPU -o {} "
        "-lstdc++  -L{}/build/ -ltaichi_lang"
        "{}",
        compiler_name(), input, gcc_opt_flag(), get_project_fn(), output,
        get_repo_dir(), extra_flags);
  } else {
    cmd = fmt::format(
        "nvcc {} -std=c++14 -shared {} -Xcompiler \"-fPIC\" --use_fast_math "
        "--ptxas-options=-allow-expensive-optimizations=true,-O3 -I {}/include "
        "-ccbin {} "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -lstdc++ -L{}/build/ -ltaichi_lang "
        "-DTLANG_GPU -o {} {}",
        input, gcc_opt_flag(), get_project_fn(), "g++-6", get_repo_dir(),
        output, extra_flags);
  }

  if (!verbose) {
    cmd += fmt::format(" 2> {}.log", input);
  }
  return cmd;
}

CompileConfig::CompileConfig() {
  arch = Arch::x86_64;
  simd_width = default_simd_width(arch);
  internal_optimization = true;
  external_optimization_level = 3;
  print_ir = false;
  max_vector_width = 8;
  force_vectorized_global_load = false;
  force_vectorized_global_store = false;
#if defined(TC_PLATFORM_OSX)
  gcc_version = -1;
#else
  gcc_version = -2;  // not 7 for faster compilation
                     // Use clang for faster speed
#endif
  serial_schedule = false;
}

std::string CompileConfig::compiler_name() {
  if (gcc_version == -1) {
    return "gcc";
  } else if (gcc_version == -2) {
    return "clang-7";
  } else {
    return fmt::format("gcc-{}", gcc_version);
  }
}

std::string CompileConfig::gcc_opt_flag() {
  TC_ASSERT(0 <= external_optimization_level &&
            external_optimization_level < 5);
  if (external_optimization_level < 4) {
    return fmt::format("-O{}", external_optimization_level);
  } else
    return "-Ofast";
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
#if defined(TC_PLATFORM_LINUX)
  std::ifstream noturbo("/sys/devices/system/cpu/intel_pstate/no_turbo");
  char c;
  noturbo >> c;
  TC_WARN_IF(c != '1',
             "You seem to be running the benchmark with Intel Turboboost.");
#endif
  TC_INFO("Eigen Version {}.{}.{}", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
          EIGEN_MINOR_VERSION);
  TC_INFO("GCC   Version {}.{}.{}", __GNUC__, __GNUC_MINOR__,
          __GNUC_PATCHLEVEL__);
  // TC_INFO("NVCC  Version {}.{}.{}", __CUDACC_VER_MAJOR__,
  // __CUDACC_VER_MINOR__,
  //        __CUDACC_VER_BUILD__);
}

TC_NAMESPACE_END
