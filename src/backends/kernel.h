#pragma once

#include "base.h"
#include "../util.h"
#include "../program.h"

TLANG_NAMESPACE_BEGIN

class Program;
class KernelCodeGen : public CodeGenBase {
 public:
  Program *prog;
  Kernel *kernel;
  KernelCodeGen(const std::string &kernel_name) : CodeGenBase(kernel_name) {
  }

  virtual void generate_header() {
    emit("#define TLANG_KERNEL\n");
    if (prog->config.debug)
      emit("#define TL_DEBUG");
    emit("#include <kernel.h>\n");
    emit("#include \"{}\"", prog->layout_fn);
    emit("using namespace taichi; using namespace Tlang;");
  }

  virtual void generate_tail() {
  }

  virtual void lower() = 0;

  virtual void codegen() = 0;

  virtual FunctionType codegen_llvm() {
    TC_NOT_IMPLEMENTED;
    return nullptr;
  }

  virtual FunctionType compile(Program &prog, Kernel &kernel);
};

TLANG_NAMESPACE_END