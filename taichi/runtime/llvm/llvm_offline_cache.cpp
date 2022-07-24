#include "llvm_offline_cache.h"

#include <queue>

#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "taichi/common/version.h"
#include "taichi/ir/transforms.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi {
namespace lang {
namespace {

using Format = LlvmOfflineCache::Format;
constexpr char kMetadataFilename[] = "metadata";

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
    const std::string &cache_file_path) {
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
  read_from_binary_file(data, tcb_path);
  return true;
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
      TI_ASSERT(data.module);
    }
    res.compiled_data_list.emplace_back(data.tasks,
                                        llvm::CloneModule(*data.module));
  }

  kernel_data.last_used_at = std::time(nullptr);

  res.created_at = kernel_data.created_at;
  res.last_used_at = kernel_data.last_used_at;
  res.kernel_key = key;
  res.args = kernel_data.args;
  return true;
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
    if (!ret) {
      err.print(filename.c_str(), llvm::errs());
      TI_ERROR("Fail to parse {}: {}", filename, err.getMessage().str());
    }
    return ret;
  }
  TI_ERROR("Unknown LLVM format={}", format_);
  return nullptr;
}

void LlvmOfflineCacheFileWriter::dump(const std::string &path,
                                      LlvmOfflineCache::Format format,
                                      bool merge_with_old) {
  taichi::create_directories(path);
  std::size_t new_kernels_size = 0;  // bytes

  for (auto &[k, v] : data_.kernels) {
    std::size_t size = 0;  // bytes
    std::string filename_prefix = taichi::join_path(path, k);

    auto write_llvm_module =
        [&filename_prefix](
            const std::string &suffix,
            std::function<void(llvm::raw_os_ostream & os)> writer) {
          const std::string filename = filename_prefix + suffix;
          std::ofstream os(filename, std::ios::out | std::ios::binary);
          TI_ERROR_IF(!os.is_open(), "File {} open failed", filename);
          llvm::raw_os_ostream llvm_os{os};
          writer(llvm_os);
          return llvm_os.tell();
        };
    {
      mangle_offloaded_task_name(k, v.compiled_data_list);
      for (int i = 0; i < v.compiled_data_list.size(); i++) {
        auto &data = v.compiled_data_list[i];
        auto *mod = data.module.get();
        TI_ASSERT(mod != nullptr);
        std::string suffix = "." + std::to_string(i);
        if (format & Format::LL) {
          size += write_llvm_module(suffix + ".ll",
                                    [mod](llvm::raw_os_ostream &os) {
                                      mod->print(os, /*AAW=*/nullptr);
                                    });
        }
        if (format & Format::BC) {
          size += write_llvm_module(suffix + ".bc",
                                    [mod](llvm::raw_os_ostream &os) {
                                      llvm::WriteBitcodeToFile(*mod, os);
                                    });
        }
      }
    }

    // Set meta info
    TI_ASSERT(v.created_at);
    TI_ASSERT(v.last_used_at);
    v.size = size;
    TI_ASSERT(v.compiled_data_list.empty() || v.size > 0);
    new_kernels_size += v.size;
  }

  data_.version[0] = TI_VERSION_MAJOR;
  data_.version[1] = TI_VERSION_MINOR;
  data_.version[2] = TI_VERSION_PATCH;
  data_.size = new_kernels_size;
  // Merge with old metadata
  if (merge_with_old) {
    LlvmOfflineCache old_data;
    if (LlvmOfflineCacheFileReader::load_meta_data(old_data, path)) {
      merge_with(std::move(old_data));
    }
  }
  // Dump metadata
  write_to_binary_file(data_, get_llvm_cache_metadata_file_path(path));
  // For debugging
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
    std::size_t cnt = 0;
    for (auto &e : compiled_data_list) {
      for (auto &offload : e.tasks) {
        std::string mangled_name = kernel_key + std::to_string(cnt++);
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
  if (policy == (std::size_t)NotClean) {
    return;
  }

  LlvmOfflineCache cache_data;
  LlvmOfflineCacheFileReader::load_meta_data(cache_data, path);

  if ((policy & CleanOldVersion) &&
      !is_current_llvm_cache_version(cache_data.version)) {
    if (bool ok = taichi::remove(get_llvm_cache_metadata_file_path(path)) &&
                  taichi::remove(get_llvm_cache_metadata_json_file_path(path));
        ok) {
      for (const auto &[k, v] : cache_data.kernels) {
        const auto files = get_possible_llvm_cache_filename_by_key(k);
        for (const auto &f : files) {
          taichi::remove(taichi::join_path(path, f));
        }
      }
    }
    return;
  }

  if (cache_data.size < max_bytes ||
      static_cast<std::size_t>(cleaning_factor * cache_data.kernels.size()) ==
          0) {
    return;
  }

  // LRU or FIFO
  using KerData = LlvmOfflineCache::KernelCacheData;
  using Comparator = std::function<bool(const KerData &, const KerData &)>;
  using PriQueue =
      std::priority_queue<KerData, std::vector<KerData>, Comparator>;

  Comparator cmp{nullptr};
  if (policy & CleanOldUsed) {  // LRU
    cmp = [](const KerData &a, const KerData &b) -> bool {
      return a.last_used_at < b.last_used_at;
    };
  } else if (policy & CleanOldCreated) {  // FIFO
    cmp = [](const KerData &a, const KerData &b) -> bool {
      return a.created_at < b.created_at;
    };
  }
  if (cmp) {
    PriQueue q(cmp);
    std::size_t cnt = cleaning_factor * cache_data.kernels.size();
    TI_ASSERT(cnt != 0);
    for (auto &[k, v] : cache_data.kernels) {
      if (q.size() == cnt && cmp(v, q.top()))
        q.pop();
      if (q.size() < cnt)
        q.push(std::move(v));
    }
    TI_ASSERT(q.size() <= cnt);
    while (!q.empty()) {
      for (int i = 0; i < q.top().compiled_data_list.size(); i++) {
        for (const auto &f : get_possible_llvm_cache_filename_by_key(
                 q.top().kernel_key + "." + std::to_string(i))) {
          taichi::remove(taichi::join_path(path, f));
        }
      }
      q.pop();
    }
    if (cnt == cache_data.kernels.size()) {  // Removed all
      taichi::remove(get_llvm_cache_metadata_file_path(path));
      taichi::remove(get_llvm_cache_metadata_json_file_path(path));
    }
  }
}

LlvmOfflineCacheFileWriter::CleanCachePolicy
LlvmOfflineCacheFileWriter::string_to_clean_cache_policy(
    const std::string &str) {
  if (str == "never")
    return Never;
  if (str == "version")
    return OnlyOldVersion;
  if (str == "lru")
    return LRU;
  if (str == "fifo")
    return FIFO;

  TI_WARN("Invalid CleanCachePolicy");
  return Never;
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
