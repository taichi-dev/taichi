#pragma once

#include "taichi/inc/constants.h"
#include "taichi/lang_util.h"
#include "taichi/backends/opengl/struct_opengl.h"
#include "taichi/runtime/opengl/opengl_api.h"

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
                const StructCompiledResult *struct_compiled,
                bool allows_nv_shader_ext)
      : kernel_name_(kernel_name),
        struct_compiled_(struct_compiled),
        allows_nv_shader_ext_(allows_nv_shader_ext) {
  }

  CompiledTaichiKernel compile(Kernel &kernel);

 private:
  void lower();
  CompiledTaichiKernel gen();

  const std::string kernel_name_;
  [[maybe_unused]] const StructCompiledResult *struct_compiled_;

  Kernel *kernel_;
  const bool allows_nv_shader_ext_;
};

}  // namespace opengl
TLANG_NAMESPACE_END
