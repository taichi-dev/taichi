#ifdef TI_WITH_DX11

#include "taichi/backends/dx/dx_program.h"

#include "taichi/backends/dx/dx_device.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"

namespace taichi {
namespace lang {
namespace directx11 {

FunctionType compile_to_executable(Kernel *kernel,
                                   gfx::GfxRuntime *runtime,
                                   gfx::SNodeTreeManager *snode_tree_mgr) {
  auto handle = runtime->register_taichi_kernel(
      std::move(gfx::run_codegen(kernel, runtime->get_ti_device(),
                                 snode_tree_mgr->get_compiled_structs())));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace directx11

Dx11ProgramImpl::Dx11ProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

FunctionType Dx11ProgramImpl::compile(Kernel *kernel,
                                      OffloadedStmt *offloaded) {
  spirv::lower(kernel);
  return directx11::compile_to_executable(kernel, runtime_.get(),
                                          snode_tree_mgr_.get());
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = std::make_shared<directx11::Dx11Device>();

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = device_.get();
  runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(runtime_.get());
}

void Dx11ProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  if (runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void Dx11ProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                             uint64 *result_buffer) {
  snode_tree_mgr_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> Dx11ProgramImpl::make_aot_module_builder() {
  if (runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(), Arch::dx11);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, Arch::dx11);
  }
}

DeviceAllocation Dx11ProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  return get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false});
}

std::unique_ptr<aot::Kernel> Dx11ProgramImpl::make_aot_kernel(Kernel &kernel) {
  spirv::lower(&kernel);
  std::vector<gfx::CompiledSNodeStructs> compiled_structs;
  gfx::GfxRuntime::RegisterParams kparams =
      gfx::run_codegen(&kernel, get_compute_device(), compiled_structs);
  return std::make_unique<gfx::KernelImpl>(runtime_.get(), std::move(kparams));
}

}  // namespace lang
}  // namespace taichi

#endif
