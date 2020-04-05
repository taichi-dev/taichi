#pragma once

#include <vector>
#include "opengl_kernel_util.h"

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct GLSLLauncherImpl;

struct GLSLLaunchGuard {
  GLSLLauncherImpl *impl;
  const std::vector<IOV> &iov;
  GLSLLaunchGuard(GLSLLauncherImpl *impl, const std::vector<IOV> &iov);
  ~GLSLLaunchGuard();
};

struct GLSLLauncher {
  std::unique_ptr<GLSLLauncherImpl> impl;
  GLSLLauncher(size_t size);
  ~GLSLLauncher();
  void begin_glsl_kernels(const std::vector<IOV> &iov);
  void end_glsl_kernels(const std::vector<IOV> &iov);
  GLSLLaunchGuard create_launch_guard(const std::vector<IOV> &iov) {
    return GLSLLaunchGuard(impl.get(), iov);
  }
};

}

TLANG_NAMESPACE_END
