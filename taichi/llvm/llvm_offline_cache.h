#pragma once

#include "taichi/common/core.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/util/io.h"

namespace taichi {
namespace lang {

struct LlvmOfflineCache {
  struct KernelCacheData {
    std::string kernel_key;
    std::unique_ptr<llvm::Module> owned_module{nullptr};
    llvm::Module *module{nullptr};
    std::vector<std::string> offloaded_task_name_list;

    KernelCacheData() = default;
    KernelCacheData(KernelCacheData &&) = default;
    KernelCacheData &operator=(KernelCacheData &&) = default;
    ~KernelCacheData() = default;
  };

  std::unordered_map<std::string, KernelCacheData> kernels;
};

class LlvmOfflineCacheFileReader {
 public:
  LlvmOfflineCacheFileReader(const std::string &path) : path_(path) {
  }

  bool get_kernel_cache(LlvmOfflineCache::KernelCacheData &res,
                        const std::string &key,
                        llvm::LLVMContext &llvm_ctx);

 private:
  std::string path_;
};

class LlvmOfflineCacheFileWriter {
 public:
  LlvmOfflineCacheFileWriter(const std::string &path) : path_(path) {
    taichi::create_directories(path);
  }

  void set_data(LlvmOfflineCache &&data) {
    this->mangled_ = false;
    this->data_ = std::move(data);
  }

  void add_kernel_cache(const std::string &key,
                        LlvmOfflineCache::KernelCacheData &&kernel_cache) {
    data_.kernels[key] = std::move(kernel_cache);
  }

  void dump();

 private:
  void mangle_offloaded_task_name(
      const std::string &kernel_key,
      llvm::Module *module,
      std::vector<std::string> &offloaded_task_name_list);

  std::string path_;
  LlvmOfflineCache data_;
  bool mangled_{false};
};

}  // namespace lang
}  // namespace taichi
