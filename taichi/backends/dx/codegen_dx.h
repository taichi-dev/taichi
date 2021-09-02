#pragma once

#include "taichi/inc/constants.h"
#include "taichi/lang_util.h"
#include "directx_api.h"
#include "struct_directx.h"

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN
namespace dx {

class DxCodeGen {
public:
  DxCodeGen(const std::string &kernel_name,
            StructCompiledResult *struct_compiled,
            HLSLLauncher *launcher)
      : struct_compiled_(struct_compiled), kernel_launcher_(launcher) {
  }  // DUMMY
  FunctionType Compile(Program* program, Kernel* kernel);

  // Copied from opengl::OpenglCodeGen
  StructCompiledResult *struct_compiled_;
  HLSLLauncher *kernel_launcher_;
};

}
TLANG_NAMESPACE_END