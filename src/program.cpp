#include <taichi/common/task.h>
#include "program.h"
#include "snode.h"
#include "backends/struct.h"
#include "backends/cpu.h"
// #include "backends/struct.h"

TLANG_NAMESPACE_BEGIN

Program *current_program = nullptr;
SNode root;

FunctionType Program::compile(Kernel &kernel) {
  FunctionType ret = nullptr;
  if (config.arch == Arch::x86_64) {
    CPUCodeGen codegen;
    ret = codegen.compile(*this, kernel);
  } else if (config.arch == Arch::gpu) {
    TC_NOT_IMPLEMENTED
    // GPUCodeGen backends;
    // function = backends.get(*this);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  return ret;
}

void Program::materialize_layout() {
  StructCompiler scomp;
  scomp.run(root);
  layout_fn = scomp.get_source_path();
  data_structure = scomp.creator();
}

TLANG_NAMESPACE_END
