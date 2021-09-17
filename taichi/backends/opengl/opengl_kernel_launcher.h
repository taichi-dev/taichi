#pragma once

#include "taichi/lang_util.h"
#include "taichi/backends/device.h"

#include <vector>

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct CompiledProgram;
struct GLSLLauncherImpl;
struct GLSLLauncher;
class GLBuffer;

struct GLSLLauncher {
  std::unique_ptr<GLSLLauncherImpl> impl;
  std::unique_ptr<Device> device{nullptr};
  GLSLLauncher(size_t size);
  ~GLSLLauncher();
  void keep(std::unique_ptr<CompiledProgram> program);

  void *result_buffer;
};

using SNodeId = std::string;

struct SNodeInfo {
  size_t stride;
  size_t length;
  std::vector<size_t> children_offsets;
  size_t elem_stride;
};

struct StructCompiledResult {
  // Source code of the SNode data structures compiled to GLSL
  std::unordered_map<SNodeId, SNodeInfo> snode_map;
  // Root buffer size in bytes.
  size_t root_size;
  std::string root_snode_type_name;
};

}  // namespace opengl

TLANG_NAMESPACE_END
