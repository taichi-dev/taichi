#pragma once

#include "base.h"
#include "../util.h"
#include "../program.h"

TLANG_NAMESPACE_BEGIN

class Program;
class KernelCodeGen : public CodeGenBase {
 public:
  Program *prog;
  Program::Kernel *kernel;
  KernelCodeGen() : CodeGenBase() {
  }

  virtual void generate_header() {
    emit("#define TLANG_KERNEL\n");
    emit("#include <kernel.h>\n");
    emit("#include \"{}\"", prog->layout_fn);
    emit("using namespace taichi; using namespace Tlang;");
  }

  virtual void generate_tail() {
  }

  virtual void lower() = 0;

  virtual void codegen() = 0;

  virtual FunctionType compile(Program &prog, Program::Kernel &kernel);
};

TLANG_NAMESPACE_END