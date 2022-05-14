#include "llvm_offline_cache.h"

#include <sstream>

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include "taichi/ir/transforms.h"
#include "taichi/llvm/llvm_context.h"

namespace taichi {
namespace lang {
namespace {
using Format = LlvmOfflineCache::Format;
}  // namespace

LlvmOfflineCacheFileReader::LlvmOfflineCacheFileReader(
    const std::string &path,
    LlvmOfflineCache::Format format)
    : path_(path), format_(format) {
  std::stringstream metafile_path_ss;
  metafile_path_ss << path_ << "/" << kMetadataFilename << ".tcb";
  read_from_binary_file(data_, metafile_path_ss.str());
}

bool LlvmOfflineCacheFileReader::get_kernel_cache(
    LlvmOfflineCache::KernelCacheData &res,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) {
  auto itr = data_.kernels.find(key);
  if (itr == data_.kernels.end()) {
    TI_ERROR("Cannot find kernel={}", key);
    return false;
  }

  auto &kernel_data = itr->second;
  if (kernel_data.owned_module == nullptr) {
    const std::string filename_prefix = path_ + "/" + key;
    kernel_data.owned_module = load_module(filename_prefix, key, llvm_ctx);
    TI_ASSERT(kernel_data.owned_module != nullptr);
    kernel_data.module = kernel_data.owned_module.get();
  }

  res.kernel_key = key;
  res.owned_module = llvm::CloneModule(*kernel_data.module);
  res.module = res.owned_module.get();
  res.offloaded_task_list = kernel_data.offloaded_task_list;
  return true;
}

std::unique_ptr<llvm::Module> LlvmOfflineCacheFileReader::load_module(
    const std::string &path_prefix,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) const {
  if (format_ & Format::BC) {
    LlvmModuleBitcodeLoader loader;
    return loader.set_bitcode_path(path_prefix + ".bc")
        .set_buffer_id(key)
        .set_inline_funcs(false)
        .load(&llvm_ctx);
  } else if (format_ & Format::LL) {
    const std::string filename = path_prefix + ".ll";
    llvm::SMDiagnostic err;
    return llvm::parseAssemblyFile(filename, err, llvm_ctx);
  }
  TI_ERROR("Unknown LLVM format={}", format_);
  return nullptr;
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
  }
  {
    std::stringstream prefix_ss;
    prefix_ss << path << "/" << kMetadataFilename;
    const std::string file_prefix = prefix_ss.str();
    write_to_binary_file(data_, file_prefix + ".tcb");
    // For debugging
    TextSerializer ts;
    ts.serialize_to_json("cache", data_);
    ts.write_to_file(file_prefix + ".json");
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
