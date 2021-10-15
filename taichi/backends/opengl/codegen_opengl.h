#pragma once

#include "taichi/inc/constants.h"
#include "taichi/lang_util.h"
#include "taichi/backends/opengl/struct_opengl.h"

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
                OpenGlRuntime *launcher)
      : kernel_name_(kernel_name),
        struct_compiled_(struct_compiled),
        runtime_(launcher) {
  }

  CompiledProgram compile(Kernel &kernel);

 private:
  void lower();
  CompiledProgram gen();

  const std::string kernel_name_;

  Kernel *kernel_;
  [[maybe_unused]] StructCompiledResult *struct_compiled_;
  [[maybe_unused]] OpenGlRuntime *runtime_;
};

}  // namespace opengl
TLANG_NAMESPACE_END
