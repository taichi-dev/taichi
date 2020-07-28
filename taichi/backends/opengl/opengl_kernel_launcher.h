#pragma once

#include <vector>

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct CompiledProgram;
struct GLSLLauncherImpl;
struct GLSLLauncher;
struct GLBuffer;

struct GLSLLauncher {
  std::unique_ptr<GLSLLauncherImpl> impl;
  GLSLLauncher(size_t size);
  ~GLSLLauncher();

  void keep(std::unique_ptr<CompiledProgram> program);
};

}  // namespace opengl

TLANG_NAMESPACE_END
