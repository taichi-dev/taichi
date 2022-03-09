#pragma once

#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"

#include "taichi/common/core.h"
#include "taichi/common/logging.h"
#include "taichi/program/compile_config.h"
#include "taichi/ir/ir.h"

TLANG_NAMESPACE_BEGIN
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

 private:
  std::string path_;
};

class LlvmOfflineCacheFileWriter {
 public:
  LlvmOfflineCacheFileWriter(const std::string &path) : path_(path) {
  }

  void set_data(LlvmOfflineCache &&data) {
    this->mangled_ = false;
    this->data_ = std::move(data);
  }

  void add_kernel_cache(const std::string &key,
                        LlvmOfflineCache::KernelCacheData &&kernel_cache) {
    data_.kernels[key] = std::move(kernel_cache);
  }

  void dump() {
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

 private:
  void mangle_offloaded_task_name(
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

  std::string path_;
  LlvmOfflineCache data_;
  bool mangled_{false};
};

TLANG_NAMESPACE_END
