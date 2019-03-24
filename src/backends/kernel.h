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
  KernelCodeGen() : CodeGenBase() {
  }

  virtual void generate_header() {
    emit("#include <common.h>\n");
    emit("#define TLANG_KERNEL\n");
    emit("#include \"{}\"", prog->layout_fn);
    emit("using namespace taichi; using namespace Tlang;");
  }

  virtual void generate_tail() {
  }

  virtual void lower() = 0;

  virtual void codegen() = 0;

  virtual FunctionType compile(Program &prog, Kernel &kernel);
};

TLANG_NAMESPACE_END