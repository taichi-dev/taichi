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

class FieldImpl : public aot::Field {
 public:
  explicit FieldImpl(const LlvmOfflineCache::FieldCacheData &field)
      : field_(field) {
  }

  explicit FieldImpl(LlvmOfflineCache::FieldCacheData &&field)
      : field_(std::move(field)) {
  }

  LlvmOfflineCache::FieldCacheData get_field() const {
    return field_;
  }

 private:
  LlvmOfflineCache::FieldCacheData field_;
};

}  // namespace

LlvmOfflineCache::KernelCacheData LlvmAotModule::load_kernel_from_cache(
    const std::string &name) {
  TI_ASSERT(cache_reader_ != nullptr);
  auto *tlctx = program_->get_llvm_context(program_->config->arch);
  LlvmOfflineCache::KernelCacheData loaded;
  auto ok = cache_reader_->get_kernel_cache(loaded, name,
                                            *tlctx->get_this_thread_context());
  TI_ERROR_IF(!ok, "Failed to load kernel={}", name);
  return loaded;
}

std::unique_ptr<aot::Kernel> LlvmAotModule::make_new_kernel(
    const std::string &name) {
  auto loaded = load_kernel_from_cache(name);
  auto fn = convert_module_to_function(name, std::move(loaded));
  return std::make_unique<KernelImpl>(fn);
}

std::unique_ptr<aot::Field> LlvmAotModule::make_new_field(
    const std::string &name) {
  // Check if "name" represents snode_tree_id.
  // Avoid using std::atoi due to its poor error handling.
  char *end;
  int snode_tree_id = static_cast<int>(strtol(name.c_str(), &end, 10 /*base*/));

  TI_ASSERT(end != name.c_str());
  TI_ASSERT(*end == '\0');

  // Load FieldCache
  LlvmOfflineCache::FieldCacheData loaded;
  auto ok = cache_reader_->get_field_cache(loaded, snode_tree_id);
  TI_ERROR_IF(!ok, "Failed to load field with id={}", snode_tree_id);

  return std::make_unique<FieldImpl>(std::move(loaded));
}

void finalize_aot_field(aot::Module *aot_module,
                        aot::Field *aot_field,
                        uint64 *result_buffer) {
  auto *llvm_aot_module = dynamic_cast<LlvmAotModule *>(aot_module);
  auto *aot_field_impl = dynamic_cast<FieldImpl *>(aot_field);

  TI_ASSERT(llvm_aot_module != nullptr);
  TI_ASSERT(aot_field_impl != nullptr);

  auto *llvm_prog = llvm_aot_module->get_program();
  const auto &field_cache = aot_field_impl->get_field();

  int snode_tree_id = field_cache.tree_id;
  if (!llvm_aot_module->is_snode_tree_initialized(snode_tree_id)) {
    llvm_prog->initialize_llvm_runtime_snodes(field_cache, result_buffer);
    llvm_aot_module->set_initialized_snode_tree(snode_tree_id);
  }
}

}  // namespace lang
}  // namespace taichi
