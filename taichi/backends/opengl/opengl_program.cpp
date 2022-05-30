#include "taichi/backends/opengl/opengl_program.h"
#include "taichi/backends/opengl/aot_module_builder_impl.h"
using namespace taichi::lang::opengl;

namespace taichi {
namespace lang {

FunctionType OpenglProgramImpl::compile(Kernel *kernel,
                                        OffloadedStmt *offloaded) {
#ifdef TI_WITH_OPENGL
  opengl::OpenglCodeGen codegen(kernel->name, &opengl_struct_compiled_.value(),
                                config->allow_nv_shader_extension);
  auto ptr = opengl_runtime_->keep(codegen.compile(*kernel));

  return [ptr, kernel, runtime = opengl_runtime_.get()](RuntimeContext &ctx) {
    ptr->launch(ctx, kernel, runtime);
  };
#else
  return [](RuntimeContext &ctx) {};
#endif
}

void OpenglProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                            KernelProfilerBase *profiler,
                                            uint64 **result_buffer_ptr) {
#ifdef TI_WITH_OPENGL
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);
  opengl_runtime_ = std::make_unique<opengl::OpenGlRuntime>();
  opengl_runtime_->result_buffer = *result_buffer_ptr;
#else
  TI_NOT_IMPLEMENTED;
#endif
}
DeviceAllocation OpenglProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  // FIXME: Why is host R/W set to true?
  return opengl_runtime_->device->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/true,
       /*export_sharing=*/false});
}

void OpenglProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  // TODO: support materializing multiple snode trees
  opengl::OpenglStructCompiler scomp;
  opengl_struct_compiled_ = scomp.run(*(tree->root()));
  TI_TRACE("OpenGL root buffer size: {} B", opengl_struct_compiled_->root_size);
}

void OpenglProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                               uint64 *result_buffer) {
#ifdef TI_WITH_OPENGL
  compile_snode_tree_types(tree);
  opengl_runtime_->add_snode_tree(opengl_struct_compiled_->root_size);
#else
  TI_NOT_IMPLEMENTED;
#endif
}

std::unique_ptr<AotModuleBuilder> OpenglProgramImpl::make_aot_module_builder() {
  // TODO: Remove this compilation guard -- AOT is a compile-time thing, so it's
  // fine to JIT to GLSL on systems without the OpenGL runtime.
#ifdef TI_WITH_OPENGL
  return std::make_unique<AotModuleBuilderImpl>(
      opengl_struct_compiled_.value(), config->allow_nv_shader_extension);
#else
  TI_NOT_IMPLEMENTED;
  return nullptr;
#endif
}

}  // namespace lang
}  // namespace taichi
