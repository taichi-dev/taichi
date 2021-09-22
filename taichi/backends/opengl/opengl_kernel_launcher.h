#pragma once

#include "taichi/lang_util.h"
#include "taichi/backends/device.h"

#include <vector>

TLANG_NAMESPACE_BEGIN

namespace opengl {

struct CompiledProgram;
struct OpenGLRuntimeImpl;
struct OpenGLRuntime;
class GLBuffer;

struct OpenGLRuntime {
  std::unique_ptr<OpenGLRuntimeImpl> impl;
  std::unique_ptr<Device> device{nullptr};
  OpenGLRuntime();
  ~OpenGLRuntime();
  void keep(std::unique_ptr<CompiledProgram> program);
  // FIXME: Currently GLSL codegen only supports single root
  void add_snode_tree(size_t size);

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
