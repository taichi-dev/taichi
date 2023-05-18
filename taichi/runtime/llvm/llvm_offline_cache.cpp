#include "llvm_offline_cache.h"

#include <queue>

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/common/cleanup.h"
#include "taichi/common/version.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/util/io.h"
#include "taichi/util/lock.h"
#include "taichi/util/offline_cache.h"

namespace taichi::lang {
namespace {

using Format = LlvmOfflineCache::Format;
constexpr char kMetadataFilename[] = "metadata";
constexpr char kMetadataFileLockName[] = "metadata.lock";

static std::string get_llvm_cache_metadata_file_path(const std::string &dir) {
  return taichi::join_path(dir, std::string(kMetadataFilename) + ".tcb");
}

static std::string get_llvm_cache_metadata_json_file_path(
    const std::string &dir) {
  return taichi::join_path(dir, std::string(kMetadataFilename) + ".json");
}

static std::vector<std::string> get_possible_llvm_cache_filename_by_key(
    const std::string &key) {
  return {
      key + "." + offline_cache::kLlvmCacheFilenameLLExt,
      key + "." + offline_cache::kLlvmCacheFilenameBCExt,
  };
}

}  // namespace

namespace offline_cache {

template <>
struct CacheCleanerUtils<LlvmOfflineCache> {
  using MetadataType = LlvmOfflineCache;
  using KernelMetaData = typename MetadataType::KernelMetadata;

  // To save metadata as file
  static bool save_metadata(const CacheCleanerConfig &config,
                            const MetadataType &data) {
    write_to_binary_file(
        data, taichi::join_path(config.path, config.metadata_filename));
    return true;
  }

  static bool save_debugging_metadata(const CacheCleanerConfig &config,
                                      const MetadataType &data) {
    TextSerializer ts;
    ts.serialize_to_json("cache", data);
    ts.write_to_file(
        taichi::join_path(config.path, config.debugging_metadata_filename));
    return true;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const CacheCleanerConfig &config,
      const KernelMetaData &kernel_meta) {
    std::vector<std::string> result;
    for (const auto &f :
         get_possible_llvm_cache_filename_by_key(kernel_meta.kernel_key)) {
      result.push_back(f);
    }
    return result;
  }

  // To remove other files except cache files and offline cache metadta files
  static void remove_other_files(const CacheCleanerConfig &config) {
    // Do nothing
  }

  // To check if a file is cache file
  static bool is_valid_cache_file(const CacheCleanerConfig &config,
                                  const std::string &name) {
    std::string ext = filename_extension(name);
    return ext == kLlvmCacheFilenameLLExt || ext == kLlvmCacheFilenameBCExt;
  }
};

}  // namespace offline_cache

// static
std::unique_ptr<LlvmOfflineCacheFileReader> LlvmOfflineCacheFileReader::make(
    const std::string &path,
    LlvmOfflineCache::Format format) {
  LlvmOfflineCache data;
  if (!load_meta_data(data, path)) {
    return nullptr;
  }
  return std::unique_ptr<LlvmOfflineCacheFileReader>(
      new LlvmOfflineCacheFileReader(path, std::move(data), format));
}

bool LlvmOfflineCacheFileReader::load_meta_data(
    LlvmOfflineCache &data,
    const std::string &cache_file_path,
    bool with_lock) {
  using offline_cache::load_metadata_with_checking;
  using Error = offline_cache::LoadMetadataError;
  const auto tcb_path = get_llvm_cache_metadata_file_path(cache_file_path);

  if (!taichi::path_exists(tcb_path)) {
    TI_DEBUG("File {} not found", tcb_path);
    return false;
  }

  if (!with_lock) {
    return Error::kNoError == load_metadata_with_checking(data, tcb_path);
  }

  std::string lock_path =
      taichi::join_path(cache_file_path, kMetadataFileLockName);
  if (lock_with_file(lock_path)) {
    auto _ = make_cleanup([&lock_path]() {
      if (!unlock_with_file(lock_path)) {
        TI_WARN(
            "Unlock {} failed. You can remove this .lock file manually and try "
            "again.",
            lock_path);
      }
    });
    return Error::kNoError == load_metadata_with_checking(data, tcb_path);
  }
  TI_WARN("Lock {} failed. You can run 'ti cache clean -p {}' and try again.",
          lock_path, cache_file_path);
  return false;
}

LlvmOfflineCacheFileReader::LlvmOfflineCacheFileReader(
    const std::string &path,
    LlvmOfflineCache &&data,
    LlvmOfflineCache::Format format)
    : path_(path), data_(std::move(data)), format_(format) {
}

size_t LlvmOfflineCacheFileReader::get_num_snode_trees() {
  return data_.fields.size();
}

bool LlvmOfflineCacheFileReader::get_field_cache(
    LlvmOfflineCache::FieldCacheData &res,
    int snode_tree_id) {
  auto itr = data_.fields.find(snode_tree_id);
  if (itr == data_.fields.end()) {
    TI_DEBUG("Cannot find field with snode_tree_id={}", snode_tree_id);
    return false;
  }

  const auto &loaded_field_cache = itr->second;
  res = loaded_field_cache;  // copy assign
  return true;
}

bool LlvmOfflineCacheFileReader::get_kernel_cache(
    LlvmOfflineCache::KernelCacheData &res,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) {
  TI_AUTO_PROF;
  auto itr = data_.kernels.find(key);
  if (itr == data_.kernels.end()) {
    TI_DEBUG("Cannot find kernel={}", key);
    return false;
  }

  auto &kernel_data = itr->second;
  auto &data = kernel_data.compiled_data;
  if (!data.module) {
    std::string filename_prefix = taichi::join_path(path_, key);
    data.module = load_module(filename_prefix, key, llvm_ctx);
    if (!data.module) {
      data_.kernels.erase(itr);
      return false;  // Must return
    }
  }
  kernel_data.last_used_at = std::time(nullptr);
  res = kernel_data.clone();

  // Verify the `res: LlvmOfflineCache::KernelCacheData`
  const auto &compiled_data = res.compiled_data;
  const auto &tasks = compiled_data.tasks;
  bool verified = true;
  for (const auto &t : tasks) {
    if (compiled_data.module->getFunction(t.name) == nullptr) {
      verified = false;
    }
  }
  if (!verified) {
    for (const auto &f : get_possible_llvm_cache_filename_by_key(key)) {
      taichi::remove(taichi::join_path(path_, f));
    }
  }

  return verified;
}

std::unique_ptr<llvm::Module> LlvmOfflineCacheFileReader::load_module(
    const std::string &path_prefix,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) const {
  TI_AUTO_PROF;
  if (format_ & Format::BC) {
    LlvmModuleBitcodeLoader loader;
    return loader
        .set_bitcode_path(path_prefix + "." +
                          offline_cache::kLlvmCacheFilenameBCExt)
        .set_buffer_id(key)
        .set_inline_funcs(false)
        .load(&llvm_ctx);
  } else if (format_ & Format::LL) {
    const std::string filename =
        path_prefix + "." + offline_cache::kLlvmCacheFilenameLLExt;
    llvm::SMDiagnostic err;
    auto ret = llvm::parseAssemblyFile(filename, err, llvm_ctx);
    if (!ret) {  // File not found or Parse failed
      TI_DEBUG("Fail to parse {}: {}", filename, err.getMessage().str());
      return nullptr;
    }
    return ret;
  }
  TI_ERROR("Unknown LLVM format={}", format_);
  return nullptr;
}

void LlvmOfflineCacheFileWriter::dump(const std::string &path,
                                      LlvmOfflineCache::Format format,
                                      bool merge_with_old) {
  auto write_llvm_module =
      [](const std::string &filename,
         std::function<void(llvm::raw_os_ostream & os)> writer) {
        std::ofstream os(filename, std::ios::out | std::ios::binary);
        TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
        llvm::raw_os_ostream llvm_os{os};
        writer(llvm_os);
        return llvm_os.tell();
      };

  using Iter = typename decltype(data_.kernels)::iterator;
  taichi::create_directories(path);
  std::size_t new_kernels_size = 0;   // bytes
  std::vector<Iter> iters_to_erased;  // Kernels which have been saved

  for (auto iter = data_.kernels.begin(); iter != data_.kernels.end(); ++iter) {
    auto &[k, v] = *iter;
    std::size_t size = 0;  // bytes
    std::string filename_prefix = taichi::join_path(path, k);
    {
      mangle_offloaded_task_name(k, v.compiled_data);
      auto &data = v.compiled_data;
      auto *mod = data.module.get();
      TI_ASSERT(mod != nullptr);
      if (format & Format::LL) {
        std::string filename =
            filename_prefix + "." + offline_cache::kLlvmCacheFilenameLLExt;
        if (!merge_with_old || try_lock_with_file(filename)) {
          size += write_llvm_module(filename, [mod](llvm::raw_os_ostream &os) {
            mod->print(os, /*AAW=*/nullptr);
          });
        } else {
          TI_DEBUG("Cache file {} exists", filename);
        }
      }
      if (format & Format::BC) {
        std::string filename =
            filename_prefix + "." + offline_cache::kLlvmCacheFilenameBCExt;
        if (!merge_with_old || try_lock_with_file(filename)) {
          size += write_llvm_module(filename, [mod](llvm::raw_os_ostream &os) {
            llvm::WriteBitcodeToFile(*mod, os);
          });
        } else {
          TI_DEBUG("Cache file {} exists", filename);
        }
      }
    }

    // Set meta info
    TI_ASSERT(v.created_at);
    TI_ASSERT(v.last_used_at);
    v.size = size;
    new_kernels_size += v.size;

    if (v.size == 0) {  // The kernel cache has been saved
      iters_to_erased.push_back(iter);
    }
  }

  // Erase the kernels which aren't needed to re-saved
  for (auto &iter : iters_to_erased) {
    data_.kernels.erase(iter);
  }

  data_.version[0] = TI_VERSION_MAJOR;
  data_.version[1] = TI_VERSION_MINOR;
  data_.version[2] = TI_VERSION_PATCH;
  data_.size = new_kernels_size;

  {
    // Lock
    // TODO(PGZXB): High overhead (read -> merge -> write). Redesign the
    // metadata file format to reduce overhead.
    std::string lock_path = taichi::join_path(path, kMetadataFileLockName);
    if (!lock_with_file(lock_path)) {
      TI_WARN(
          "Lock {} failed. You can run 'ti cache clean -p {}' and try again.",
          lock_path, path);
      return;
    }
    auto _ = make_cleanup([&lock_path]() {
      if (!unlock_with_file(lock_path)) {
        TI_WARN(
            "Unlock {} failed. You can remove this .lock file manually and try "
            "again.",
            lock_path);
      }
    });

    // Merge with old metadata
    if (merge_with_old) {
      LlvmOfflineCache old_data;
      if (LlvmOfflineCacheFileReader::load_meta_data(old_data, path, false)) {
        merge_with(std::move(old_data));
      }
    }

    // Dump metadata
    std::string target_path = get_llvm_cache_metadata_file_path(path);
    write_to_binary_file(data_, target_path);
  }
  // For debugging (Not safe: without locking)
  TextSerializer ts;
  ts.serialize_to_json("cache", data_);
  ts.write_to_file(get_llvm_cache_metadata_json_file_path(path));
}

void LlvmOfflineCacheFileWriter::merge_with(LlvmOfflineCache &&data) {
  // Note: merge this->data_ with data, new cover old
  auto &new_kernels = data_.kernels;
  auto &new_fields = data_.fields;
  auto &old_kernels = data.kernels;
  auto &old_fields = data.fields;

  for (auto &[k, v] : new_fields) {
    old_fields[k] = std::move(v);
  }
  for (auto &[k, v] : new_kernels) {
    auto iter = old_kernels.find(k);
    if (iter == old_kernels.end()) {
      data.size += v.size;
      old_kernels[k] = std::move(v);
    } else {
      data.size += v.size - iter->second.size;
      iter->second = std::move(v);
    }
  }

  data_ = std::move(data);
}

void LlvmOfflineCacheFileWriter::mangle_offloaded_task_name(
    const std::string &kernel_key,
    LLVMCompiledKernel &compiled_data) {
  if (!mangled_) {
    for (auto &offload : compiled_data.tasks) {
      std::string mangled_name =
          offline_cache::mangle_name(offload.name, kernel_key);
      auto func = compiled_data.module->getFunction(offload.name);
      TI_ASSERT(func != nullptr);
      func->setName(mangled_name);
      offload.name = mangled_name;
    }
  }
}

void LlvmOfflineCacheFileWriter::clean_cache(const std::string &path,
                                             CleanCachePolicy policy,
                                             int max_bytes,
                                             double cleaning_factor) {
  using CacheCleaner = offline_cache::CacheCleaner<LlvmOfflineCache>;
  offline_cache::CacheCleanerConfig config;
  config.path = path;
  config.policy = policy;
  config.cleaning_factor = cleaning_factor;
  config.max_size = max_bytes;
  config.metadata_filename = std::string(kMetadataFilename) + ".tcb";
  config.debugging_metadata_filename = std::string(kMetadataFilename) + ".json";
  config.metadata_lock_name = kMetadataFileLockName;
  CacheCleaner::run(config);
}

LlvmOfflineCache::KernelCacheData LlvmOfflineCache::KernelCacheData::clone()
    const {
  LlvmOfflineCache::KernelCacheData result;
  result.kernel_key = kernel_key;
  result.args = args;
  result.rets = rets;
  result.compiled_data = compiled_data.clone();
  result.size = size;
  result.created_at = created_at;
  result.last_used_at = last_used_at;
  result.ret_size = ret_size;
  result.ret_type = ret_type;
  result.args_size = args_size;
  result.args_type = args_type;
  return result;
}

LLVM::CompiledKernelData::InternalData
LlvmOfflineCache::KernelCacheData::convert_to_llvm_ckd_data() const {
  LLVM::CompiledKernelData::InternalData result;
  result.args = args;
  result.rets = rets;
  result.compiled_data = compiled_data.clone();
  result.ret_size = ret_size;
  result.ret_type = ret_type;
  result.args_size = args_size;
  result.args_type = args_type;
  return result;
}

}  // namespace taichi::lang
