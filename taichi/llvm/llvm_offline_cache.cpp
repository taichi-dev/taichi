#include "llvm_offline_cache.h"

#include <sstream>

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Module.h"

#include "taichi/ir/transforms.h"
#include "taichi/llvm/llvm_context.h"

namespace taichi {
namespace lang {
namespace {
using Format = LlvmOfflineCache::Format;
}  // namespace

bool LlvmOfflineCacheFileReader::get_kernel_cache(
    LlvmOfflineCache::KernelCacheData &res,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) {
  res.kernel_key = key;
  const std::string filename_prefix = path_ + "/" + key;
  if (format_ & Format::BC) {
    LlvmModuleBitcodeLoader loader;
    res.owned_module = loader.set_bitcode_path(filename_prefix + ".bc")
                           .set_buffer_id(key)
                           .set_inline_funcs(false)
                           .load(&llvm_ctx);
  } else if (format_ & Format::LL) {
    const std::string filename = filename_prefix + ".ll";
    llvm::SMDiagnostic err;
    res.owned_module = llvm::parseAssemblyFile(filename, err, llvm_ctx);
  } else {
    TI_ERROR("Unknown LLVM format={}", format_);
    return false;
  }

  res.module = res.owned_module.get();
  if (!res.module) {
    return false;
  }

  {
    const std::string filename = filename_prefix + "_otnl.txt";
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

void LlvmOfflineCacheFileWriter::dump(const std::string &path,
                                      LlvmOfflineCache::Format format) {
  taichi::create_directories(path);
  for (auto &[k, v] : data_.kernels) {
    std::stringstream filename_ss;
    filename_ss << path << "/" << k;
    std::string filename_prefix = filename_ss.str();

    auto write_llvm_module =
        [&filename_prefix](
            const std::string &suffix,
            std::function<void(llvm::raw_os_ostream & os)> writer) {
          const std::string filename = filename_prefix + suffix;
          std::ofstream os(filename, std::ios::out | std::ios::binary);
          TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
          llvm::raw_os_ostream llvm_os{os};
          writer(llvm_os);
        };
    {
      auto *mod = v.module;
      if (!mod) {
        mod = v.owned_module.get();
      }
      TI_ASSERT(mod != nullptr);

      mangle_offloaded_task_name(k, mod, v.offloaded_task_list);
      if (format & Format::LL) {
        write_llvm_module(".ll", [mod](llvm::raw_os_ostream &os) {
          mod->print(os, /*AAW=*/nullptr);
        });
      }
      if (format & Format::BC) {
        write_llvm_module(".bc", [mod](llvm::raw_os_ostream &os) {
          llvm::WriteBitcodeToFile(*mod, os);
        });
      }
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
