
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
  if (arch == Arch::x64) {
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

