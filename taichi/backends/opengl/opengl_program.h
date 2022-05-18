#pragma once

#include "taichi/backends/opengl/struct_opengl.h"

#include "taichi/runtime/opengl/opengl_kernel_launcher.h"
#include "taichi/runtime/opengl/opengl_api.h"
#include "taichi/backends/opengl/codegen_opengl.h"

#include "taichi/system/memory_pool.h"
#include "taichi/common/logging.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program_impl.h"

#include <optional>

namespace taichi {
namespace lang {

class OpenglProgramImpl : public ProgramImpl {
 public:
  OpenglProgramImpl(CompileConfig &config) : ProgramImpl(config) {
  }
  FunctionType compile(Kernel *kernel, OffloadedStmt *offloaded) override;

  std::size_t get_snode_num_dynamically_allocated(
      SNode *snode,
      uint64 *result_buffer) override {
    return 0;  // TODO: support dynamic snode alloc in vulkan
  }

  void materialize_runtime(MemoryPool *memory_pool,
                           KernelProfilerBase *profiler,
                           uint64 **result_buffer_ptr) override;

  void compile_snode_tree_types(SNodeTree *tree) override;

  void materialize_snode_tree(SNodeTree *tree, uint64 *result_buffer) override;

  void synchronize() override {
  }

  DeviceAllocation allocate_memory_ndarray(std::size_t alloc_size,
                                           uint64 *result_buffer) override;

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override;

  void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_NOT_IMPLEMENTED
  }

  ~OpenglProgramImpl() override {
  }

 private:
  std::optional<opengl::StructCompiledResult> opengl_struct_compiled_;
  std::unique_ptr<opengl::OpenGlRuntime> opengl_runtime_;
};

}  // namespace lang
}  // namespace taichi
