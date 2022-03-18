#include "llvm_offline_cache.h"

#include "llvm/AsmParser/Parser.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Module.h"

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
      res.offloaded_task_name_list.push_back(std::move(line));
    }
  }
  return true;
}

void LlvmOfflineCacheFileWriter::dump() {
  for (auto &[k, v] : data_.kernels) {
    std::string filename_prefix = path_ + "/" + k;
    {
      std::string filename = filename_prefix + ".ll";
      std::ofstream os(filename, std::ios::out | std::ios::binary);
      TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
      llvm::SMDiagnostic err;
      llvm::LLVMContext ctx;
      llvm::raw_os_ostream llvm_os(os);
      if (v.module) {
        mangle_offloaded_task_name(k, v.module, v.offloaded_task_name_list);
        v.module->print(llvm_os, nullptr);
      } else if (v.owned_module) {
        mangle_offloaded_task_name(k, v.owned_module.get(),
                                   v.offloaded_task_name_list);
        v.owned_module->print(llvm_os, nullptr);
      } else
        TI_ASSERT(false);
    }
    {
      std::string filename = filename_prefix + "_otnl.txt";
      std::ofstream os(filename, std::ios::out | std::ios::binary);
      TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
      for (const auto &name : v.offloaded_task_name_list) {
        os << name << '\n';
      }
    }
  }
}

void LlvmOfflineCacheFileWriter::mangle_offloaded_task_name(
    const std::string &kernel_key,
    llvm::Module *module,
    std::vector<std::string> &offloaded_task_name_list) {
  if (!mangled_) {
    std::size_t cnt = 0;
    for (auto &e : offloaded_task_name_list) {
      std::string mangled_name = kernel_key + std::to_string(cnt++);
      auto func = module->getFunction(e);
      TI_ASSERT(func != nullptr);
      func->setName(mangled_name);
      e = mangled_name;
    }
  }
}

}  // namespace lang
}  // namespace taichi
