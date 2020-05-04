#pragma once

#include <vector>
#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct CompiledProgram;
struct GLSLLauncherImpl;

struct GLSLLaunchGuard {
  GLSLLauncherImpl *impl;
  const std::vector<IOV> &iov;
  GLSLLaunchGuard(GLSLLauncherImpl *impl, const std::vector<IOV> &iov);
  ~GLSLLaunchGuard();
  void *map_buffer(size_t idx);
  void unmap_buffer(size_t idx);
};

struct GLSLLauncher {
  std::unique_ptr<GLSLLauncherImpl> impl;
  GLSLLauncher(size_t size);
  ~GLSLLauncher();
  GLSLLaunchGuard create_launch_guard(const std::vector<IOV> &iov) {
    return GLSLLaunchGuard(impl.get(), iov);
  }

  void keep(std::unique_ptr<CompiledProgram> program);
};

}  // namespace opengl

TLANG_NAMESPACE_END
