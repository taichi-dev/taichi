#pragma once

#include "taichi/inc/constants.h"
#include "taichi/lang_util.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN
namespace opengl {

class OpenglCodeGen {
 public:
  OpenglCodeGen(const std::string &kernel_name,
                StructCompiledResult *struct_compiled,
                GLSLLauncher *launcher)
      : kernel_name_(kernel_name),
        struct_compiled_(struct_compiled),
        kernel_launcher_(launcher) {
  }

  FunctionType compile(Program &program, Kernel &kernel);

 private:
  void lower();
  FunctionType gen();

  const std::string kernel_name_;

  Program *prog_;
  Kernel *kernel_;
  [[maybe_unused]] StructCompiledResult *struct_compiled_;
  [[maybe_unused]] GLSLLauncher *kernel_launcher_;
  size_t global_tmps_buffer_size_{0};
};

}  // namespace opengl
TLANG_NAMESPACE_END
