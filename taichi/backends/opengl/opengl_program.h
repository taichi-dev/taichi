#pragma once

#include "taichi/backends/vulkan/runtime.h"
#include "taichi/backends/opengl/opengl_api.h"

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

  void compile_snode_tree_types(SNodeTree *tree) override {
    TI_NOT_IMPLEMENTED
  }

  void materialize_snode_tree(
      SNodeTree *tree,
      std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
      uint64 *result_buffer) override;

  void synchronize() override {
  }

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder() override {
    TI_NOT_IMPLEMENTED
  }

  virtual void destroy_snode_tree(SNodeTree *snode_tree) override {
    TI_NOT_IMPLEMENTED
  }

  ~OpenglProgramImpl() override;

 private:
  std::unique_ptr<Device> device_;
  std::unique_ptr<vulkan::VkRuntime> runtime_;
};

bool is_opengl_api_available();

}  // namespace lang
}  // namespace taichi
