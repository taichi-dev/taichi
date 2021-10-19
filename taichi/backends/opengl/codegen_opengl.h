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
                const StructCompiledResult *struct_compiled)
      : kernel_name_(kernel_name), struct_compiled_(struct_compiled) {
  }

  CompiledProgram compile(Kernel &kernel);

 private:
  void lower();
  CompiledProgram gen();

  const std::string kernel_name_;
  [[maybe_unused]] const StructCompiledResult *struct_compiled_;

  Kernel *kernel_;
};

}  // namespace opengl
TLANG_NAMESPACE_END
