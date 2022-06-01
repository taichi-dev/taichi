#include "taichi/backends/cpu/aot_module_loader_impl.h"
#include "taichi/llvm/llvm_aot_module_loader.h"

#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/codegen/codegen_llvm.h"

namespace taichi {
namespace lang {
namespace {

class AotModuleImpl : public LlvmAotModule {
 public:
  explicit AotModuleImpl(const cpu::AotModuleParams &params)
      : LlvmAotModule(params.module_path, params.program) {
  }

  Arch arch() const override {
    return Arch::x64;
  }

 private:
  FunctionType convert_module_to_function(
      const std::string &name,
      LlvmOfflineCache::KernelCacheData &&loaded) override {
    auto *tlctx = program_->get_llvm_context(program_->config->arch);

    const auto &tasks = loaded.offloaded_task_list;
    std::vector<OffloadedTask> offloaded_tasks;
    offloaded_tasks.reserve(tasks.size());
    for (const auto &t : tasks) {
      OffloadedTask ot{/*codegen=*/nullptr};
      ot.name = t.name;
      ot.block_dim = t.block_dim;
      ot.grid_dim = t.grid_dim;
      offloaded_tasks.push_back(std::move(ot));
    }

    ModuleToFunctionConverter converter{tlctx, program_};
    return converter.convert(name, loaded.args, std::move(loaded.owned_module),
                             std::move(offloaded_tasks));
  }

  std::unique_ptr<aot::KernelTemplate> make_new_kernel_template(
      const std::string &name) override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }

  std::unique_ptr<aot::Field> make_new_field(const std::string &name) override {
    TI_NOT_IMPLEMENTED;
    return nullptr;
  }
};

}  // namespace

namespace cpu {

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params) {
  auto mod = std::make_unique<AotModuleImpl>(
      std::any_cast<const AotModuleParams &>(mod_params));
  return mod;
}

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
