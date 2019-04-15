#include "kernel.h"
#include <taichi/system/timer.h>
#include <xxhash.h>

TLANG_NAMESPACE_BEGIN

FunctionType KernelCodeGen::compile(taichi::Tlang::Program &prog,
                                    taichi::Tlang::Kernel &kernel) {
  this->prog = &kernel.program;
  this->kernel = &kernel;
  lower();
  codegen();
  write_source();
  auto t = Time::get_time();
  auto compiler_config = get_current_program().config.compiler_config();
  auto pp_fn = get_source_path() + ".i";
  auto preprocess_cmd =
      get_current_program().config.preprocess_cmd(get_source_path(), pp_fn);
  std::system(preprocess_cmd.c_str());
  std::ifstream ifs(pp_fn);
  TC_ASSERT(ifs);
  auto hash_input =
      compiler_config + std::string(std::istreambuf_iterator<char>(ifs),
                                    std::istreambuf_iterator<char>());
  auto hash = XXH64(hash_input.data(), hash_input.size(), 0);

  std::string cached_binary_fn = db_folder() + fmt::format("/{}.so", hash);
  std::ifstream key_file(cached_binary_fn);
  if (key_file) {
    std::system(
        fmt::format("cp {} {}", cached_binary_fn, get_library_path()).c_str());
  } else {
    auto cmd = get_current_program().config.compile_cmd(get_source_path(),
                                                        get_library_path());
    auto compile_ret = std::system(cmd.c_str());
    if (compile_ret != 0) {
      auto cmd = get_current_program().config.compile_cmd(
          get_source_path(), get_library_path(), true);
      trash(std::system(cmd.c_str()));
      TC_ERROR("Source {} compilation failed.", get_source_path());
    } else {
      std::system(fmt::format("cp {} {}", get_library_path(), cached_binary_fn)
                      .c_str());
    }
  }
  disassemble();
  return load_function();
}

TLANG_NAMESPACE_END
