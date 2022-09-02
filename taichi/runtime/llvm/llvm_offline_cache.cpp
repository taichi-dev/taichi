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

namespace taichi {
namespace lang {
namespace {

using Format = LlvmOfflineCache::Format;
constexpr char kMetadataFilename[] = "metadata";
constexpr char kMetadataFileLockName[] = "metadata.lock";

static bool is_current_llvm_cache_version(
    const LlvmOfflineCache::Version &ver) {
  // TODO(PGZXB): Do more detailed checking
  return ver[0] == TI_VERSION_MAJOR && ver[1] == TI_VERSION_MINOR &&
         ver[2] == TI_VERSION_PATCH;
}

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
      key + ".ll",
      key + ".bc",
  };
}

}  // namespace

namespace offline_cache {

template <>
struct CacheCleanerUtils<LlvmOfflineCache> {
  using MetadataType = LlvmOfflineCache;
  using KernelMetaData = typename MetadataType::KernelMetadata;

  // To load metadata from file
  static bool load_metadata(MetadataType &result, const std::string &filepath) {
    return read_from_binary_file(result, filepath);
  }

  // To save metadata as file
  static bool save_metadata(const MetadataType &data,
                            const std::string &filepath) {
    write_to_binary_file(data, filepath);
    return true;
  }

  static bool save_debugging_metadata(const MetadataType &data,
                                      const std::string &filepath) {
    TextSerializer ts;
    ts.serialize_to_json("cache", data);
    ts.write_to_file(get_llvm_cache_metadata_json_file_path(filepath));
    return true;
  }

  // To check version
  static bool check_version(const Version &version) {
    return is_current_llvm_cache_version(version);
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const KernelMetaData &kernel_meta) {
    std::vector<std::string> result;
    for (int i = 0; i < kernel_meta.compiled_data_list.size(); i++) {
      for (const auto &f : get_possible_llvm_cache_filename_by_key(
               kernel_meta.kernel_key + "." + std::to_string(i))) {
        result.push_back(f);
      }
    }
    return result;
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
  const auto tcb_path = get_llvm_cache_metadata_file_path(cache_file_path);
  {
    // No the best way to check for filepath existence, but whatever... See
    // https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exists-using-standard-c-c11-14-17-c
    std::ifstream fs(tcb_path, std::ios::in | std::ios::binary);
    if (!fs.good()) {
      TI_DEBUG("LLVM cache {} does not exist", cache_file_path);
      return false;
    }
  }

  if (!with_lock) {
    read_from_binary_file(data, tcb_path);
    return true;
  }

  std::string lock_path =
      taichi::join_path(cache_file_path, kMetadataFileLockName);
  if (lock_with_file(lock_path)) {
    auto _ = make_cleanup([&lock_path]() {
      if (!unlock_with_file(lock_path)) {
        TI_WARN("Unlock {} failed", lock_path);
      }
    });
    read_from_binary_file(data, tcb_path);
    return true;
  }
  TI_WARN("Lock {} failed", lock_path);
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
  for (int i = 0; i < kernel_data.compiled_data_list.size(); i++) {
    auto &data = kernel_data.compiled_data_list[i];
    if (!data.module) {
      std::string filename_prefix =
          taichi::join_path(path_, key + "." + std::to_string(i));
      data.module = load_module(filename_prefix, key, llvm_ctx);
      if (!data.module) {
        data_.kernels.erase(itr);
        return false;  // Must return
      }
    }
    res.compiled_data_list.emplace_back(
        data.tasks, llvm::CloneModule(*data.module), data.used_tree_ids,
        data.struct_for_tls_sizes);
  }

  kernel_data.last_used_at = std::time(nullptr);

  res.created_at = kernel_data.created_at;
  res.last_used_at = kernel_data.last_used_at;
  res.kernel_key = key;
  res.args = kernel_data.args;

  // Verify the `res: LlvmOfflineCache::KernelCacheData`
  bool verified_all = true;
  const auto &compiled_data_list = res.compiled_data_list;
  for (std::size_t i = 0; i < compiled_data_list.size(); ++i) {
    const auto &data = compiled_data_list[i];
    const auto &tasks = data.tasks;
    bool verified = true;
    for (const auto &t : tasks) {
      if (data.module->getFunction(t.name) == nullptr) {
        verified = false;
        verified_all = false;
      }
    }
    if (!verified) {
      for (const auto &f : get_possible_llvm_cache_filename_by_key(
               key + "." + std::to_string(i))) {
        taichi::remove(taichi::join_path(path_, f));
      }
    }
  }

  return verified_all;
}

std::unique_ptr<llvm::Module> LlvmOfflineCacheFileReader::load_module(
    const std::string &path_prefix,
    const std::string &key,
    llvm::LLVMContext &llvm_ctx) const {
  TI_AUTO_PROF;
  if (format_ & Format::BC) {
    LlvmModuleBitcodeLoader loader;
    return loader.set_bitcode_path(path_prefix + ".bc")
        .set_buffer_id(key)
        .set_inline_funcs(false)
        .load(&llvm_ctx);
  } else if (format_ & Format::LL) {
    const std::string filename = path_prefix + ".ll";
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
      mangle_offloaded_task_name(k, v.compiled_data_list);
      for (int i = 0; i < v.compiled_data_list.size(); i++) {
        auto &data = v.compiled_data_list[i];
        auto *mod = data.module.get();
        TI_ASSERT(mod != nullptr);
        std::string suffix = "." + std::to_string(i);
        if (format & Format::LL) {
          std::string filename = filename_prefix + suffix + ".ll";
          if (try_lock_with_file(filename)) {  // Not exists
            size +=
                write_llvm_module(filename, [mod](llvm::raw_os_ostream &os) {
                  mod->print(os, /*AAW=*/nullptr);
                });
          } else {
            TI_DEBUG("Cache file {} exists", filename);
          }
        }
        if (format & Format::BC) {
          std::string filename = filename_prefix + suffix + ".bc";
          if (try_lock_with_file(filename)) {  // Not exists
            size +=
                write_llvm_module(filename, [mod](llvm::raw_os_ostream &os) {
                  llvm::WriteBitcodeToFile(*mod, os);
                });
          } else {
            TI_DEBUG("Cache file {} exists", filename);
          }
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
      TI_WARN("Lock {} failed", lock_path);
      return;
    }
    auto _ = make_cleanup([&lock_path]() {
      if (!unlock_with_file(lock_path)) {
        TI_WARN("Unlock {} failed", lock_path);
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
    std::vector<LLVMCompiledData> &compiled_data_list) {
  if (!mangled_) {
    for (auto &e : compiled_data_list) {
      for (auto &offload : e.tasks) {
        std::string mangled_name =
            offline_cache::mangle_name(offload.name, kernel_key);
        TI_DEBUG(
            "Mangle offloaded-task from internal name '{}' to offline cache "
            "key '{}'",
            offload.name, mangled_name);
        auto func = e.module->getFunction(offload.name);
        TI_ASSERT(func != nullptr);
        func->setName(mangled_name);
        offload.name = mangled_name;
      }
    }
  }
}

void LlvmOfflineCacheFileWriter::clean_cache(const std::string &path,
                                             CleanCachePolicy policy,
                                             int max_bytes,
                                             double cleaning_factor) {
  using CacheCleaner = offline_cache::CacheCleaner<LlvmOfflineCache>;
  CacheCleaner::Params params;
  params.path = path;
  params.policy = policy;
  params.cleaning_factor = cleaning_factor;
  params.max_size = max_bytes;
  params.metadata_filename = std::string(kMetadataFilename) + ".tcb";
  params.debugging_metadata_filename = std::string(kMetadataFilename) + ".json";
  params.metadata_lock_name = kMetadataFileLockName;
  CacheCleaner::run(params);
}

LlvmOfflineCache::KernelCacheData LlvmOfflineCache::KernelCacheData::clone()
    const {
  std::vector<LLVMCompiledData> new_data_list;
  for (const auto &data : compiled_data_list) {
    new_data_list.push_back(data.clone());
  }
  return {kernel_key, args, std::move(new_data_list)};
}
}  // namespace lang
}  // namespace taichi
