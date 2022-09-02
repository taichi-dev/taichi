#pragma once

#include <ctime>
#include <cstdint>
#include <queue>
#include <string>
#include <type_traits>
#include <unordered_map>

#include "taichi/common/core.h"
#include "taichi/common/cleanup.h"
#include "taichi/util/io.h"
#include "taichi/util/lock.h"

namespace taichi {
namespace lang {
namespace offline_cache {

using Version = std::uint16_t[3];  // {MAJOR, MINOR, PATCH}

enum CleanCacheFlags {
  NotClean = 0b000,
  CleanOldVersion = 0b001,
  CleanOldUsed = 0b010,
  CleanOldCreated = 0b100
};

enum CleanCachePolicy {
  Never = NotClean,
  OnlyOldVersion = CleanOldVersion,
  LRU = CleanOldVersion | CleanOldUsed,
  FIFO = CleanOldVersion | CleanOldCreated
};

inline CleanCachePolicy string_to_clean_cache_policy(const std::string &str) {
  if (str == "never") {
    return Never;
  } else if (str == "version") {
    return OnlyOldVersion;
  } else if (str == "lru") {
    return LRU;
  } else if (str == "fifo") {
    return FIFO;
  }
  return Never;
}

template <typename MetadataType>
struct CacheCleanerUtils {
  using KernelMetaData = typename MetadataType::KernelMetadata;

  // To load metadata from file
  static bool load_metadata(MetadataType &result, const std::string &filepath) {
    TI_NOT_IMPLEMENTED;
  }

  // To save metadata as file
  static bool save_metadata(const MetadataType &data,
                            const std::string &filepath) {
    TI_NOT_IMPLEMENTED;
  }

  static bool save_debugging_metadata(const MetadataType &data,
                                      const std::string &filepath) {
    TI_NOT_IMPLEMENTED;
  }

  // To check version
  static bool check_version(const Version &version) {
    TI_NOT_IMPLEMENTED;
  }

  // To get cache files name
  static std::vector<std::string> get_cache_files(
      const KernelMetaData &kernel_meta) {
    TI_NOT_IMPLEMENTED;
  }
};

template <typename MetadataType>
class CacheCleaner {
  using Utils = CacheCleanerUtils<MetadataType>;
  using KernelMetadata = typename MetadataType::KernelMetadata;

 public:
  struct Params {
    std::string path;
    CleanCachePolicy policy;
    int max_size{0};
    double cleaning_factor{0.f};
    std::string metadata_filename;
    std::string debugging_metadata_filename;
    std::string metadata_lock_name;
  };

  static void run(const Params &config) {
    TI_ASSERT(!config.path.empty());
    TI_ASSERT(config.max_size > 0);
    TI_ASSERT(!config.metadata_filename.empty());
    TI_ASSERT(!config.metadata_lock_name.empty());
    const auto policy = config.policy;
    const auto &path = config.path;
    const auto metadata_file =
        taichi::join_path(path, config.metadata_filename);
    const auto debugging_metadata_file =
        taichi::join_path(path, config.debugging_metadata_filename);

    if (policy == (std::size_t)NotClean) {
      return;
    }
    if (!taichi::path_exists(path)) {
      return;
    }

    MetadataType cache_data;
    std::vector<std::string> files_to_rm;
    bool ok_rm_meta = false;

    // 1. Remove/Update metadata files
    {
      std::string lock_path =
          taichi::join_path(path, config.metadata_lock_name);
      if (!lock_with_file(lock_path)) {
        TI_WARN("Lock {} failed", lock_path);
        return;
      }
      auto _ = make_cleanup([&lock_path]() {
        TI_DEBUG("Stop cleaning cache");
        if (!unlock_with_file(lock_path)) {
          TI_WARN("Unlock {} failed", lock_path);
        }
      });
      TI_DEBUG("Start cleaning cache");

      if (!Utils::load_metadata(cache_data, metadata_file)) {
        return;
      }

      if ((policy & CleanOldVersion) &&
          !Utils::check_version(cache_data.version)) {
        if (taichi::remove(metadata_file)) {
          taichi::remove(debugging_metadata_file);
          for (const auto &[k, v] : cache_data.kernels) {
            for (const auto &f : Utils::get_cache_files(v)) {
              taichi::remove(taichi::join_path(path, f));
            }
          }
        }
        return;
      }

      if (cache_data.size < config.max_size ||
          static_cast<std::size_t>(config.cleaning_factor *
                                   cache_data.kernels.size()) == 0) {
        return;
      }

      // LRU or FIFO
      using KerData = std::pair<const std::string, KernelMetadata>;
      using Comparator = std::function<bool(const KerData *, const KerData *)>;
      using PriQueue =
          std::priority_queue<const KerData *, std::vector<const KerData *>,
                              Comparator>;

      Comparator cmp{nullptr};
      if (policy & CleanOldUsed) {  // LRU
        cmp = [](const KerData *a, const KerData *b) -> bool {
          return a->second.last_used_at < b->second.last_used_at;
        };
      } else if (policy & CleanOldCreated) {  // FIFO
        cmp = [](const KerData *a, const KerData *b) -> bool {
          return a->second.created_at < b->second.created_at;
        };
      }

      if (cmp) {
        PriQueue q(cmp);
        std::size_t cnt = config.cleaning_factor * cache_data.kernels.size();
        TI_ASSERT(cnt != 0);
        for (const auto &e : cache_data.kernels) {
          if (q.size() == cnt && cmp(&e, q.top())) {
            q.pop();
          }
          if (q.size() < cnt) {
            q.push(&e);
          }
        }
        TI_ASSERT(q.size() <= cnt);
        while (!q.empty()) {
          const auto *e = q.top();
          for (const auto &f : Utils::get_cache_files(e->second)) {
            files_to_rm.push_back(f);
          }
          cache_data.size -= e->second.size;
          cache_data.kernels.erase(e->first);
          q.pop();
        }

        if (cache_data.kernels.empty()) {  // Remove
          ok_rm_meta = taichi::remove(metadata_file);
          taichi::remove(debugging_metadata_file);
        } else {  // Update
          Utils::save_metadata(cache_data, metadata_file);
          ok_rm_meta = true;
        }
      }
    }

    // 2. Remove cache files
    if (ok_rm_meta) {
      if (!cache_data.kernels.empty()) {
        // For debugging (Not safe: without locking)
        Utils::save_debugging_metadata(cache_data, debugging_metadata_file);
      }
      for (const auto &f : files_to_rm) {
        auto file_path = taichi::join_path(path, f);
        taichi::remove(file_path);
      }
    }
  }
};

}  // namespace offline_cache
}  // namespace lang
}  // namespace taichi
