#pragma once

#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/program/kernel.h"
#include "taichi/util/io.h"

#include "llvm/IR/Module.h"

namespace taichi {
namespace lang {

struct LlvmOfflineCache {
  enum Format {
    LL = 0x01,
    BC = 0x10,
  };

  struct OffloadedTaskCacheData {
    std::string name;
    int block_dim{0};
    int grid_dim{0};

    TI_IO_DEF(name, block_dim, grid_dim);
  };

  struct KernelCacheData {
    std::string kernel_key;
    std::vector<OffloadedTaskCacheData> offloaded_task_list;

    std::unique_ptr<llvm::Module> owned_module{nullptr};
    llvm::Module *module{nullptr};

    KernelCacheData() = default;
    KernelCacheData(KernelCacheData &&) = default;
    KernelCacheData &operator=(KernelCacheData &&) = default;
    ~KernelCacheData() = default;

    TI_IO_DEF(kernel_key, offloaded_task_list);
  };

  std::unordered_map<std::string, KernelCacheData> kernels;

  TI_IO_DEF(kernels);
};

class LlvmOfflineCacheFileReader {
 public:
  LlvmOfflineCacheFileReader(
      const std::string &path,
      LlvmOfflineCache::Format format = LlvmOfflineCache::Format::LL);

  bool get_kernel_cache(LlvmOfflineCache::KernelCacheData &res,
                        const std::string &key,
                        llvm::LLVMContext &llvm_ctx);

 private:
  std::unique_ptr<llvm::Module> load_module(const std::string &path_prefix,
                                            const std::string &key,
                                            llvm::LLVMContext &llvm_ctx) const;

  std::string path_;
  LlvmOfflineCache::Format format_;
  LlvmOfflineCache data_;
};

class LlvmOfflineCacheFileWriter {
 public:
  void set_data(LlvmOfflineCache &&data) {
    this->mangled_ = false;
    this->data_ = std::move(data);
  }

  void add_kernel_cache(const std::string &key,
                        LlvmOfflineCache::KernelCacheData &&kernel_cache) {
    data_.kernels[key] = std::move(kernel_cache);
  }

  void dump(const std::string &path,
            LlvmOfflineCache::Format format = LlvmOfflineCache::Format::LL);

  void set_no_mangle() {
    mangled_ = true;
  }

 private:
  void mangle_offloaded_task_name(
      const std::string &kernel_key,
      llvm::Module *module,
      std::vector<LlvmOfflineCache::OffloadedTaskCacheData>
          &offloaded_task_list);

  LlvmOfflineCache data_;
  bool mangled_{false};
};

}  // namespace lang
}  // namespace taichi
