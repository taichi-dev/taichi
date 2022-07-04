#include "taichi/runtime/cuda/aot_module_loader_impl.h"
#include "taichi/runtime/llvm/llvm_aot_module_loader.h"

#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/cuda/codegen_cuda.h"

namespace taichi {
namespace lang {
namespace {

class AotModuleImpl : public LlvmAotModule {
 public:
  explicit AotModuleImpl(const cuda::AotModuleParams &params)
      : LlvmAotModule(params.module_path, params.program) {
  }

 private:
  FunctionType convert_module_to_function(
      const std::string &name,
      LlvmOfflineCache::KernelCacheData &&loaded) override {
    Arch arch = program_->config->arch;
    TI_ASSERT(arch == Arch::cuda);
    auto *tlctx = program_->get_llvm_context(arch);

    CUDAModuleToFunctionConverter converter{tlctx, program_};
    return converter.convert(name, loaded.args,
                             std::move(loaded.compiled_data_list));
  }

  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &name) override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }
};

}  // namespace

namespace cuda {

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params) {
  auto mod = std::make_unique<AotModuleImpl>(
      std::any_cast<const AotModuleParams &>(mod_params));
  return mod;
}

}  // namespace cuda
}  // namespace lang
}  // namespace taichi
