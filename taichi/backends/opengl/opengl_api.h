#pragma once

#include "taichi/common/util.h"

#include <string>
#include <vector>

#include "opengl_kernel_launcher.h"

TLANG_NAMESPACE_BEGIN

class Kernel;

namespace opengl {

void initialize_opengl();
bool is_opengl_api_available();
int opengl_get_threads_per_group();
extern bool opengl_has_GL_NV_shader_atomic_float;

struct CompiledProgram {
  struct Impl;
  std::unique_ptr<Impl> impl;

  // disscussion:
  // https://github.com/taichi-dev/taichi/pull/696#issuecomment-609332527
  CompiledProgram(CompiledProgram &&) = default;
  CompiledProgram &operator=(CompiledProgram &&) = default;

  CompiledProgram(Kernel *kernel, size_t gtmp_size);
  ~CompiledProgram();

  void add(const std::string &kernel_name,
           const std::string &kernel_source_code,
           int num_groups,
           const UsedFeature &used);
  void launch(Context &ctx, GLSLLauncher *launcher) const;
};

}  // namespace opengl

TLANG_NAMESPACE_END
