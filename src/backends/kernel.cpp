#include "kernel.h"

TLANG_NAMESPACE_BEGIN

FunctionType KernelCodeGen::compile(taichi::Tlang::Program &prog,
                                    taichi::Tlang::Kernel &kernel) {
  this->prog = &kernel.program;
  this->kernel = &kernel;
  lower();
  codegen();
  //std::string source = get_source();
  write_source();
  auto cmd = get_current_program().config.compile_cmd(get_source_path(),
                                                      get_library_path());
  auto compile_ret = std::system(cmd.c_str());
  if (compile_ret != 0) {
    auto cmd = get_current_program().config.compile_cmd(
        get_source_path(), get_library_path(), true);
    trash(std::system(cmd.c_str()));
    TC_ERROR("Source {} compilation failed.", get_source_path());
  }
  disassemble();
  return load_function();
}

TLANG_NAMESPACE_END
