#include "taichi/backends/opengl/opengl_program.h"
using namespace taichi::lang::opengl;

namespace taichi {
namespace lang {

FunctionType OpenglProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
  opengl::OpenglCodeGen codegen(kernel->name, &opengl_struct_compiled_.value(),
                                opengl_kernel_launcher_.get());
  return codegen.compile(*kernel);
}

void OpenglProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  // doesn't do anything other than alloc result buffer. runtime is materialized
  // together with snode tree.
  // TODO: separate runtime materialization and tree materialization.
}

void OpenglProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    std::unordered_map<int, SNode *> &,
    SNodeGlobalVarExprMap &,
    uint64 *result_buffer) {
  // TODO: support materializing multiple snode trees
  auto *const root = tree->root();
  opengl::OpenglStructCompiler scomp;
  opengl_struct_compiled_ = scomp.run(*root);
  TI_TRACE("OpenGL root buffer size: {} B", opengl_struct_compiled_->root_size);
  opengl_kernel_launcher_ = std::make_unique<opengl::GLSLLauncher>(
      opengl_struct_compiled_->root_size);
  opengl_kernel_launcher_->result_buffer = result_buffer;
}

}  // namespace lang
}  // namespace taichi
