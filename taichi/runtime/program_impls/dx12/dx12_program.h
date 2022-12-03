#pragma once

#ifdef TI_WITH_DX12

#include "taichi/runtime/program_impls/llvm/llvm_program.h"

namespace taichi {
namespace lang {

class Dx12ProgramImpl : public LlvmProgramImpl {
 public:
  Dx12ProgramImpl(CompileConfig &config);

  std::unique_ptr<AotModuleBuilder> make_aot_module_builder(
      const DeviceCapabilityConfig &caps) override;
};

}  // namespace lang
}  // namespace taichi

#endif
