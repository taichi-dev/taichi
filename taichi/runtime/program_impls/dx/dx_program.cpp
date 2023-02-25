#ifdef TI_WITH_DX11

#include "taichi/runtime/program_impls/dx/dx_program.h"

#include "taichi/rhi/dx/dx_api.h"
#include "taichi/codegen/spirv/kernel_compiler.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/codegen/spirv/spirv_codegen.h"

namespace taichi::lang {
namespace {

FunctionType register_params_to_executable(
    gfx::GfxRuntime::RegisterParams &&params,
    gfx::GfxRuntime *runtime) {
  auto handle = runtime->register_taichi_kernel(std::move(params));
  return [runtime, handle](RuntimeContext &ctx) {
    runtime->launch_kernel(handle, &ctx);
  };
}

}  // namespace

Dx11ProgramImpl::Dx11ProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

FunctionType Dx11ProgramImpl::compile(const CompileConfig &compile_config,
                                      Kernel *kernel) {
  // NOTE: Temporary implementation
  // TODO(PGZXB): Final solution: compile -> load_or_compile + launch_kernel
  auto &mgr = get_kernel_compilation_manager();
  const auto &compiled = mgr.load_or_compile(
      compile_config, runtime_->get_ti_device()->get_caps(), *kernel);
  const auto *spirv_compiled =
      dynamic_cast<const spirv::CompiledKernelData *>(&compiled);
  const auto &spirv_data = spirv_compiled->get_internal_data();
  gfx::GfxRuntime::RegisterParams params;
  params.kernel_attribs = spirv_data.metadata.kernel_attribs;
  params.task_spirv_source_codes = spirv_data.src.spirv_src;
  params.num_snode_trees = spirv_data.metadata.num_snode_trees;
  return register_params_to_executable(std::move(params), runtime_.get());
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  *result_buffer_ptr = (uint64 *)memory_pool->allocate(
      sizeof(uint64) * taichi_result_buffer_entries, 8);

  device_ = directx11::make_dx11_device();

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

std::unique_ptr<AotModuleBuilder> Dx11ProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(), Arch::dx11, *config, caps);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, Arch::dx11, *config, caps);
  }
}

DeviceAllocation Dx11ProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  return get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false});
}

std::unique_ptr<KernelCompiler> Dx11ProgramImpl::make_kernel_compiler() {
  spirv::KernelCompiler::Config cfg;
  cfg.compiled_struct_data = runtime_ ? &snode_tree_mgr_->get_compiled_structs()
                                      : &aot_compiled_snode_structs_;
  return std::make_unique<spirv::KernelCompiler>(std::move(cfg));
}

}  // namespace taichi::lang

#endif
