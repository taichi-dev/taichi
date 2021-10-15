#include "taichi/backends/opengl/opengl_program.h"
#include "taichi/backends/opengl/aot_module_builder_impl.h"
using namespace taichi::lang::opengl;

namespace taichi {
namespace lang {

FunctionType OpenglProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
#ifdef TI_WITH_OPENGL
  opengl::OpenglCodeGen codegen(kernel->name, &opengl_struct_compiled_.value(),
                                opengl_runtime_.get());
  auto ptr = opengl_runtime_->keep(codegen.compile(*kernel));

  return [ptr, runtime = opengl_runtime_.get()](Context &ctx) {
    ptr->launch(ctx, runtime);
  };
#else
  return [](Context &ctx) {};
#endif
}

void OpenglProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
#ifdef TI_WITH_OPENGL
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  opengl_runtime_ = std::make_unique<opengl::OpenGlRuntime>();
#else
  TI_NOT_IMPLEMENTED;
#endif
}

void OpenglProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &,
    std::unordered_map<int, SNode *> &,
    uint64 *result_buffer) {
#ifdef TI_WITH_OPENGL
  // TODO: support materializing multiple snode trees
  auto *const root = tree->root();
  opengl::OpenglStructCompiler scomp;
  opengl_struct_compiled_ = scomp.run(*root);
  TI_TRACE("OpenGL root buffer size: {} B", opengl_struct_compiled_->root_size);
  opengl_runtime_->add_snode_tree(opengl_struct_compiled_->root_size);
  opengl_runtime_->result_buffer = result_buffer;
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::unique_ptr<AotModuleBuilder> OpenglProgramImpl::make_aot_module_builder() {
#ifdef TI_WITH_OPENGL
  return std::make_unique<AotModuleBuilderImpl>(opengl_struct_compiled_.value(),
                                                *opengl_runtime_);
#else
  TI_NOT_IMPLEMENTED;
  return nullptr;
#endif
}

}  // namespace lang
}  // namespace taichi
