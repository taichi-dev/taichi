#pragma once

#include "taichi/common/core.h"
#include "taichi/program/kernel.h"
#include "taichi/llvm/llvm_fwd.h"
#include "taichi/util/io.h"

namespace taichi {
namespace lang {

std::string get_offline_cache_key_of_kernel(Kernel *kernel);

struct LlvmOfflineCache {
  struct OffloadedTaskCacheData {
    std::string name;
    int block_dim{0};
    int grid_dim{0};
  };
  struct KernelCacheData {
    std::string kernel_key;
    std::unique_ptr<llvm::Module> owned_module{nullptr};
    llvm::Module *module{nullptr};
    std::vector<OffloadedTaskCacheData> offloaded_task_list;

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
      std::vector<LlvmOfflineCache::OffloadedTaskCacheData>
          &offloaded_task_list);

  std::string path_;
  LlvmOfflineCache data_;
  bool mangled_{false};
};

}  // namespace lang
}  // namespace taichi
