#include "llvm_offline_cache.h"

#include <sstream>

#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Module.h"
#include "taichi/ir/transforms.h"

namespace taichi {
namespace lang {

bool LlvmOfflineCacheFileReader::get_kernel_cache(
    LlvmOfflineCache::KernelCacheData &res,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) {
  res.kernel_key = key;
  std::string filename_prefix = path_ + "/" + key;
  {
    std::string filename = filename_prefix + ".ll";
    llvm::SMDiagnostic err;
    res.owned_module = llvm::parseAssemblyFile(filename, err, llvm_ctx);
    res.module = res.owned_module.get();
    if (!res.module)
      return false;
  }
  {
    std::string filename = filename_prefix + "_otnl.txt";
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open())
      return false;
    while (true) {
      std::string line;
      std::getline(in, line, '\n');
      if (line.empty())
        break;
      std::istringstream iss(line);
      auto &task = res.offloaded_task_list.emplace_back();
      iss >> task.name >> task.block_dim >> task.grid_dim;
    }
  }
  return true;
}

void LlvmOfflineCacheFileWriter::dump(const std::string &path) {
  taichi::create_directories(path);
  for (auto &[k, v] : data_.kernels) {
    std::stringstream filename_ss;
    filename_ss << path << "/" << k;
    std::string filename_prefix = filename_ss.str();
    {
      std::string filename = filename_prefix + ".ll";
      std::ofstream os(filename, std::ios::out | std::ios::binary);
      TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
      llvm::SMDiagnostic err;
      llvm::LLVMContext ctx;
      llvm::raw_os_ostream llvm_os(os);
      if (v.module) {
        mangle_offloaded_task_name(k, v.module, v.offloaded_task_list);
        v.module->print(llvm_os, nullptr);
      } else if (v.owned_module) {
        mangle_offloaded_task_name(k, v.owned_module.get(),
                                   v.offloaded_task_list);
        v.owned_module->print(llvm_os, nullptr);
      } else
        TI_ASSERT(false);
    }
    {
      std::string filename = filename_prefix + "_otnl.txt";
      std::ofstream os(filename, std::ios::out | std::ios::binary);
      TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
      for (const auto &task : v.offloaded_task_list) {
        os << task.name << ' ' << task.block_dim << ' ' << task.grid_dim
           << '\n';
      }
    }
  }
}

void LlvmOfflineCacheFileWriter::mangle_offloaded_task_name(
    const std::string &kernel_key,
    llvm::Module *module,
    std::vector<LlvmOfflineCache::OffloadedTaskCacheData>
        &offloaded_task_list) {
  if (!mangled_) {
    std::size_t cnt = 0;
    for (auto &e : offloaded_task_list) {
      std::string mangled_name = kernel_key + std::to_string(cnt++);
      auto func = module->getFunction(e.name);
      TI_ASSERT(func != nullptr);
      func->setName(mangled_name);
      e.name = mangled_name;
    }
  }
}

}  // namespace lang
}  // namespace taichi
