#ifdef TI_WITH_DX12

#include "taichi/runtime/program_impls/dx12/dx12_program.h"
#include "taichi/runtime/dx12/aot_module_builder_impl.h"
#include "taichi/rhi/dx12/dx12_api.h"

namespace taichi {
namespace lang {

Dx12ProgramImpl::Dx12ProgramImpl(CompileConfig &config)
    : LlvmProgramImpl(config, nullptr) {
}

std::unique_ptr<AotModuleBuilder> Dx12ProgramImpl::make_aot_module_builder(
    const DeviceCapabilityConfig &caps) {
  return std::make_unique<directx12::AotModuleBuilderImpl>(*config, this,
                                                           *get_llvm_context());
}

}  // namespace lang
}  // namespace taichi

#endif
