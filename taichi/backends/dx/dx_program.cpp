#include "taichi/backends/dx/dx_program.h"
#include "taichi/backends/dx/dx_device.h"

namespace taichi {
namespace lang {

Dx11ProgramImpl::Dx11ProgramImpl(CompileConfig &config) : ProgramImpl(config) {
}

FunctionType Dx11ProgramImpl::compile(Kernel *kernel,
                                      OffloadedStmt *offloaded) {
  TI_NOT_IMPLEMENTED;
}

void Dx11ProgramImpl::materialize_runtime(MemoryPool *memory_pool,
                                          KernelProfilerBase *profiler,
                                          uint64 **result_buffer_ptr) {
  TI_NOT_IMPLEMENTED;
}

void Dx11ProgramImpl::synchronize() {
  TI_NOT_IMPLEMENTED;
}

void Dx11ProgramImpl::materialize_snode_tree(
    SNodeTree *tree,
    std::vector<std::unique_ptr<SNodeTree>> &snode_trees_,
    uint64 *result_buffer_ptr) {
  TI_NOT_IMPLEMENTED;
}

std::unique_ptr<AotModuleBuilder> Dx11ProgramImpl::make_aot_module_builder() {
  return nullptr;
}

void Dx11ProgramImpl::destroy_snode_tree(SNodeTree *snode_tree) {
  TI_NOT_IMPLEMENTED;
}

}  // namespace lang
}  // namespace taichi
