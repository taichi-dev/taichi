#pragma once

#include <memory>

#include "llvm/IR/Module.h"
#include "taichi/common/core.h"
#include "taichi/common/serialization.h"
#include "taichi/llvm/launch_arg_info.h"
#include "taichi/program/kernel.h"
#include "taichi/util/io.h"

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
    std::vector<LlvmLaunchArgInfo> args;
    std::vector<OffloadedTaskCacheData> offloaded_task_list;

    std::unique_ptr<llvm::Module> owned_module{nullptr};
    llvm::Module *module{nullptr};

    KernelCacheData() = default;
    KernelCacheData(KernelCacheData &&) = default;
    KernelCacheData &operator=(KernelCacheData &&) = default;
    ~KernelCacheData() = default;

    TI_IO_DEF(kernel_key, args, offloaded_task_list);
  };

  struct FieldCacheData {
    struct SNodeCacheData {
      int id;
      SNodeType type = SNodeType::undefined;
      size_t cell_size_bytes{0};
      size_t chunk_size{0};

      TI_IO_DEF(id, type, cell_size_bytes, chunk_size);
    };

    int tree_id{0};
    int root_id{0};
    size_t root_size{0};
    std::vector<SNodeCacheData> snode_metas;

    TI_IO_DEF(tree_id, root_id, root_size, snode_metas);

    // TODO(zhanlue)
    //  Serialize/Deserialize the llvm::Module from StructCompiler
    //  At runtime, make sure loaded Field-Modules and Kernel-Modules are linked
    //  altogether.
  };

  // TODO(zhanlue): we need a better identifier for each FieldCacheData
  // (SNodeTree) Given that snode_tree_id is not continuous, it is ridiculous to
  // ask the users to remember each of the snode_tree_ids
  // ** Find a way to name each SNodeTree **
  std::unordered_map<int, FieldCacheData> fields;  // key = snode_tree_id

  std::unordered_map<std::string, KernelCacheData>
      kernels;  // key = kernel_name

  TI_IO_DEF(kernels);
};

class LlvmOfflineCacheFileReader {
 public:
  bool get_kernel_cache(LlvmOfflineCache::KernelCacheData &res,
                        const std::string &key,
                        llvm::LLVMContext &llvm_ctx);

  static std::unique_ptr<LlvmOfflineCacheFileReader> make(
      const std::string &path,
      LlvmOfflineCache::Format format = LlvmOfflineCache::Format::LL);

 private:
  LlvmOfflineCacheFileReader(const std::string &path,
                             LlvmOfflineCache &&data,
                             LlvmOfflineCache::Format format);

  std::unique_ptr<llvm::Module> load_module(const std::string &path_prefix,
                                            const std::string &key,
                                            llvm::LLVMContext &llvm_ctx) const;

  std::string path_;
  LlvmOfflineCache data_;
  LlvmOfflineCache::Format format_;
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
