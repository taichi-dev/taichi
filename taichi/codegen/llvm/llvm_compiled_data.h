#pragma once

#include <memory>

#include "llvm/IR/Module.h"

namespace taichi {
namespace lang {

class OffloadedTask {
 public:
  std::string name;
  int block_dim{0};
  int grid_dim{0};

  OffloadedTask(const std::string &name = "",
                int block_dim = 0,
                int grid_dim = 0)
      : name(name), block_dim(block_dim), grid_dim(grid_dim){};
  TI_IO_DEF(name, block_dim, grid_dim);
};

struct LLVMCompiledData {
  std::vector<OffloadedTask> tasks;
  std::unique_ptr<llvm::Module> module{nullptr};
  LLVMCompiledData() = default;
  LLVMCompiledData(LLVMCompiledData &&) = default;
  LLVMCompiledData(std::vector<OffloadedTask> tasks,
                   std::unique_ptr<llvm::Module> module)
      : tasks(std::move(tasks)), module(std::move(module)) {
  }
  TI_IO_DEF(tasks);
};

}  // namespace lang
}  // namespace taichi
