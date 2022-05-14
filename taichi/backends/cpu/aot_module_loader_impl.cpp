#include "taichi/backends/cpu/aot_module_loader_impl.h"

#include "taichi/llvm/llvm_offline_cache.h"
#include "taichi/llvm/llvm_program.h"
#include "taichi/codegen/codegen_llvm.h"

namespace taichi {
namespace lang {
namespace cpu {
namespace {

class KernelImpl : public aot::Kernel {
 public:
  explicit KernelImpl(FunctionType fn) : fn_(fn) {
  }

  void launch(RuntimeContext *ctx) override {
    fn_(*ctx);
  }

 private:
  FunctionType fn_;
};

class AotModuleImpl : public aot::Module {
 public:
  explicit AotModuleImpl(const AotModuleParams &params)
      : program_(params.program), cache_reader_(params.module_path) {
  }

  Arch arch() const override {
    return Arch::x64;
  }

  uint64_t version() const override {
    return 0;
  }

  size_t get_root_size() const override {
    return 0;
  }

 private:
  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override {
    auto *tlctx = program_->get_llvm_context(program_->config->arch);
    LlvmOfflineCache::KernelCacheData loaded;
    auto ok = cache_reader_.get_kernel_cache(loaded, name,
                                             *tlctx->get_this_thread_context());
    TI_ERROR_IF(!ok, "Failed to load kernel={}", name);

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
    auto fn =
        converter.convert(name, /*args=*/{}, std::move(loaded.owned_module),
                          std::move(offloaded_tasks));
    return std::make_unique<KernelImpl>(fn);
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

  LlvmProgramImpl *const program_{nullptr};
  LlvmOfflineCacheFileReader cache_reader_;
};

}  // namespace

std::unique_ptr<aot::Module> make_aot_module(std::any mod_params) {
  auto mod = std::make_unique<AotModuleImpl>(
      std::any_cast<const AotModuleParams &>(mod_params));
  return mod;
}

}  // namespace cpu
}  // namespace lang
}  // namespace taichi
