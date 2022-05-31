#include "taichi/llvm/llvm_aot_module_loader.h"

namespace taichi {
namespace lang {
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

}  // namespace

LlvmOfflineCache::KernelCacheData LLVMAotModule::load_kernel_from_cache(
    const std::string &name) {
  TI_ASSERT(cache_reader_ != nullptr);
  auto *tlctx = program_->get_llvm_context(program_->config->arch);
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = cache_reader_->get_kernel_cache(loaded, name,
                                            *tlctx->get_this_thread_context());
  TI_ERROR_IF(!ok, "Failed to load kernel={}", name);
  return loaded;
}

std::unique_ptr<aot::Kernel> LLVMAotModule::make_new_kernel(
    const std::string &name) {
  auto loaded = load_kernel_from_cache(name);
  auto fn = convert_module_to_function(name, std::move(loaded));
  return std::make_unique<KernelImpl>(fn);
}

}  // namespace lang
}  // namespace taichi
