#include "taichi/util/offline_cache.h"

namespace taichi::lang::offline_cache {

constexpr std::size_t offline_cache_key_length = 65;
constexpr std::size_t min_mangled_name_length = offline_cache_key_length + 2;

void disable_offline_cache_if_needed(CompileConfig *config) {
  TI_ASSERT(config);
  if (config->offline_cache) {
    if (config->print_preprocessed_ir || config->print_ir ||
        config->print_accessor_ir) {
      config->offline_cache = false;
      TI_WARN(
          "Disable offline_cache because print_preprocessed_ir or print_ir or "
          "print_accessor_ir is enabled");
    }
  }
}

std::string get_cache_path_by_arch(const std::string &base_path, Arch arch) {
  std::string subdir;
  if (arch_uses_llvm(arch)) {
    subdir = kLlvmCachSubPath;
  } else if (arch_uses_spirv(arch)) {
    subdir = kSpirvCacheSubPath;
  } else if (arch == Arch::dx12) {
    subdir = "dx12";
  } else {
    return base_path;
  }
  return taichi::join_path(base_path, subdir);
}

std::string mangle_name(const std::string &primal_name,
                        const std::string &key) {
  // Result: {primal_name}{key: char[65]}_{(checksum(primal_name)) ^
  // checksum(key)}
  if (key.size() != offline_cache_key_length) {
    return primal_name;
  }
  std::size_t checksum1{0}, checksum2{0};
  for (auto &e : primal_name) {
    checksum1 += std::size_t(e);
  }
  for (auto &e : key) {
    checksum2 += std::size_t(e);
  }
  return fmt::format("{}{}_{}", primal_name, key, checksum1 ^ checksum2);
}

bool try_demangle_name(const std::string &mangled_name,
                       std::string &primal_name,
                       std::string &key) {
  if (mangled_name.size() < min_mangled_name_length) {
    return false;
  }

  std::size_t checksum{0}, checksum1{0}, checksum2{0};
  auto pos = mangled_name.find_last_of('_');
  if (pos == std::string::npos) {
    return false;
  }
  try {
    checksum = std::stoull(mangled_name.substr(pos + 1));
  } catch (const std::exception &) {
    return false;
  }

  std::size_t i = 0, primal_len = pos - offline_cache_key_length;
  for (i = 0; i < primal_len; ++i) {
    checksum1 += (int)mangled_name[i];
  }
  for (; i < pos; ++i) {
    checksum2 += (int)mangled_name[i];
  }
  if ((checksum1 ^ checksum2) != checksum) {
    return false;
  }

  primal_name = mangled_name.substr(0, primal_len);
  key = mangled_name.substr(primal_len, offline_cache_key_length);
  TI_ASSERT(key.size() == offline_cache_key_length);
  TI_ASSERT(primal_name.size() + key.size() == pos);
  return true;
}

std::size_t clean_offline_cache_files(const std::string &path) {
  std::vector<const char *> sub_dirs = {kLlvmCachSubPath, kSpirvCacheSubPath,
                                        kMetalCacheSubPath};

  auto is_cache_filename = [](const std::string &name) {
    const auto ext = taichi::filename_extension(name);
    return ext == kLlvmCacheFilenameBCExt || ext == kLlvmCacheFilenameLLExt ||
           ext == kSpirvCacheFilenameExt || ext == kMetalCacheFilenameExt ||
           ext == kTiCacheFilenameExt || ext == "lock" || ext == "tcb";
  };

  std::size_t count = 0;

  // Temp implementation. We will refactor the offline cache
  taichi::traverse_directory(
      path, [&count, &sub_dirs, &is_cache_filename, &path](
                const std::string &name, bool is_dir) {
        if (is_dir) {  // ~/.cache/taichi/ticache/llvm ...
          for (auto subdir : sub_dirs) {
            auto subpath = taichi::join_path(path, subdir);

            if (taichi::path_exists(subpath)) {
              taichi::traverse_directory(
                  subpath, [&count, &is_cache_filename, &subpath](
                               const std::string &name, bool is_dir) {
                    if (is_cache_filename(name) && !is_dir) {
                      const auto fpath = taichi::join_path(subpath, name);
                      TI_TRACE("Removing {}", fpath);
                      bool ok = taichi::remove(fpath);
                      count += ok ? 1 : 0;
                      TI_WARN_IF(!ok, "Remove {} failed", fpath);
                    }
                  });
            }
          }
        } else if (is_cache_filename(name)) {
          const auto fpath = taichi::join_path(path, name);
          TI_TRACE("Removing {}", fpath);
          bool ok = taichi::remove(fpath);
          count += ok ? 1 : 0;
          TI_WARN_IF(!ok, "Remove {} failed", fpath);
        }
      });

  return count;
}

}  // namespace taichi::lang::offline_cache
