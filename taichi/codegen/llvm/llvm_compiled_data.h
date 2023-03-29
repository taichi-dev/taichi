#pragma once

#include <memory>
#include <unordered_set>

#include "llvm/IR/Module.h"
#include "taichi/common/serialization.h"

namespace taichi::lang {

class OffloadedTask {
 public:
  std::string name;
  int block_dim{0};
  int grid_dim{0};
  int dynamic_shared_array_bytes{0};

  explicit OffloadedTask(const std::string &name = "",
                         int block_dim = 0,
                         int grid_dim = 0,
                         int dynamic_shared_array_bytes = 0)
      : name(name),
        block_dim(block_dim),
        grid_dim(grid_dim),
        dynamic_shared_array_bytes(dynamic_shared_array_bytes){};
  TI_IO_DEF(name, block_dim, grid_dim, dynamic_shared_array_bytes);
};

struct LLVMCompiledTask {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> struct_for_tls_sizes;
  LLVMCompiledTask() = default;
  LLVMCompiledTask(LLVMCompiledTask &&) = default;
  LLVMCompiledTask &operator=(LLVMCompiledTask &&) = default;
  LLVMCompiledTask(std::vector<OffloadedTask> tasks,
                   std::unique_ptr<llvm::Module> module,
                   std::unordered_set<int> used_tree_ids,
                   std::unordered_set<int> struct_for_tls_sizes)
      : tasks(std::move(tasks)),
        module(std::move(module)),
        used_tree_ids(std::move(used_tree_ids)),
        struct_for_tls_sizes(std::move(struct_for_tls_sizes)) {
  }
  LLVMCompiledTask clone() const;
  TI_IO_DEF(tasks);
};

struct LLVMCompiledKernel {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  LLVMCompiledKernel() = default;
  LLVMCompiledKernel(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel &operator=(LLVMCompiledKernel &&) = default;
  LLVMCompiledKernel(std::vector<OffloadedTask> tasks,
                     std::unique_ptr<llvm::Module> module)
      : tasks(std::move(tasks)), module(std::move(module)) {
  }
  LLVMCompiledKernel clone() const;
  TI_IO_DEF(tasks);
};

}  // namespace taichi::lang
