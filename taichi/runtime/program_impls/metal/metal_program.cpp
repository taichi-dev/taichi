#include "taichi/runtime/program_impls/metal/metal_program.h"

#include "taichi/analysis/offline_cache_util.h"
#include "taichi/aot/graph_data.h"
#include "taichi/codegen/spirv/kernel_compiler.h"
#include "taichi/codegen/spirv/compiled_kernel_data.h"
#include "taichi/rhi/metal/metal_device.h"
#include "taichi/runtime/gfx/aot_module_builder_impl.h"
#include "taichi/runtime/gfx/snode_tree_manager.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/gfx/kernel_launcher.h"
#include "taichi/util/offline_cache.h"
#include "taichi/rhi/common/memory_pool.h"

namespace taichi::lang {

MetalProgramImpl::MetalProgramImpl(CompileConfig &config)
    : ProgramImpl(config) {
}

void MetalProgramImpl::materialize_runtime(KernelProfilerBase *profiler,
                                           uint64 **result_buffer_ptr) {
  *result_buffer_ptr =
      (uint64 *)MemoryPool::get_instance(config->arch)
          .allocate(sizeof(uint64) * taichi_result_buffer_entries, 8);

  embedded_device_ =
      std::unique_ptr<metal::MetalDevice>(metal::MetalDevice::create());

  gfx::GfxRuntime::Params params;
  params.host_result_buffer = *result_buffer_ptr;
  params.device = embedded_device_.get();
  gfx_runtime_ = std::make_unique<gfx::GfxRuntime>(std::move(params));
  snode_tree_mgr_ = std::make_unique<gfx::SNodeTreeManager>(gfx_runtime_.get());
}

void MetalProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  if (gfx_runtime_) {
    snode_tree_mgr_->materialize_snode_tree(tree);
  } else {
    gfx::CompiledSNodeStructs compiled_structs =
        gfx::compile_snode_structs(*tree->root());
    aot_compiled_snode_structs_.push_back(compiled_structs);
  }
}

void MetalProgramImpl::materialize_snode_tree(SNodeTree *tree,
                                              uint64 *result_buffer) {
  snode_tree_mgr_->materialize_snode_tree(tree);
}

std::unique_ptr<AotModuleBuilder> MetalProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  if (gfx_runtime_) {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        snode_tree_mgr_->get_compiled_structs(),
        get_kernel_compilation_manager(), *config, caps);
  } else {
    return std::make_unique<gfx::AotModuleBuilderImpl>(
        aot_compiled_snode_structs_, get_kernel_compilation_manager(), *config,
        caps);
  }
}

DeviceAllocation MetalProgramImpl::allocate_memory_ndarray(
    std::size_t alloc_size,
    uint64 *result_buffer) {
  DeviceAllocation alloc;
  RhiResult res = get_compute_device()->allocate_memory(
      {alloc_size, /*host_write=*/false, /*host_read=*/false,
       /*export_sharing=*/false},
      &alloc);
  TI_ASSERT(res == RhiResult::success);
  return alloc;
}

DeviceAllocation MetalProgramImpl::allocate_texture(const ImageParams &params) {
  return gfx_runtime_->create_image(params);
}

void MetalProgramImpl::enqueue_compute_op_lambda(
    std::function<void(Device *device, CommandList *cmdlist)> op,
    const std::vector<ComputeOpImageRef> &image_refs) {
  gfx_runtime_->enqueue_compute_op_lambda(op, image_refs);
}

MetalProgramImpl::~MetalProgramImpl() {
  gfx_runtime_.reset();
  embedded_device_.reset();
}

std::unique_ptr<KernelCompiler> MetalProgramImpl::make_kernel_compiler() {
  spirv::KernelCompiler::Config cfg;
  cfg.compiled_struct_data = gfx_runtime_
                                 ? &snode_tree_mgr_->get_compiled_structs()
                                 : &aot_compiled_snode_structs_;
  return std::make_unique<spirv::KernelCompiler>(std::move(cfg));
}

std::unique_ptr<KernelLauncher> MetalProgramImpl::make_kernel_launcher() {
  gfx::KernelLauncher::Config cfg;
  cfg.gfx_runtime_ = gfx_runtime_.get();
  return std::make_unique<gfx::KernelLauncher>(std::move(cfg));
}

DeviceCapabilityConfig MetalProgramImpl::get_device_caps() {
  TI_ASSERT(gfx_runtime_);
  return gfx_runtime_->get_ti_device()->get_caps();
}

}  // namespace taichi::lang
