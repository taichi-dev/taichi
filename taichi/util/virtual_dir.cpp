#include "virtual_dir.h"
#include "taichi/util/zip.h"
#include <filesystem>

namespace taichi {
namespace io {

struct FilesystemVirtualDir : public VirtualDir {
  std::string base_dir_;

  explicit FilesystemVirtualDir(const std::string &base_dir)
      : base_dir_(base_dir) {
  }

  static std::unique_ptr<VirtualDir> create(const std::string &base_dir) {
    TI_ASSERT(std::filesystem::is_directory(base_dir));

    std::string base_dir2;
    if (base_dir.empty()) {
      base_dir2 = "./";
    } else if (base_dir.back() != '/') {
      base_dir2 = base_dir + "/";
    } else {
      base_dir2 = base_dir;
    }

    return std::unique_ptr<VirtualDir>(new FilesystemVirtualDir(base_dir));
  }

  bool get_file_size(const std::string &path, size_t &size) const override {
    std::fstream f(base_dir_ + path, std::ios::in | std::ios::ate);
    if (!f.is_open()) {
      return false;
    }
    size = f.tellg();
    return true;
  }
  size_t load_file(const std::string &path,
                   void *data,
                   size_t size) const override {
    std::fstream f(base_dir_ + path, std::ios::in);
    if (!f.is_open()) {
      return false;
    }
    f.seekg(std::ios::beg);
    size_t n = f.readsome((char *)data, size);
    return n;
  }
};
struct ZipArchiveVirtualDir : public VirtualDir {
  zip::ZipArchive archive_;

  explicit ZipArchiveVirtualDir(zip::ZipArchive &&archive)
      : archive_(std::move(archive)) {
  }

  static std::unique_ptr<VirtualDir> create(const std::string &archive_path) {
    std::fstream f(archive_path, std::ios::in | std::ios::ate);
    std::vector<uint8_t> archive_data(f.tellg());
    f.seekg(std::ios::beg);
    f.read((char *)archive_data.data(), archive_data.size());

    zip::ZipArchive archive;
    bool succ = zip::ZipArchive::try_from_bytes(archive_data.data(),
                                                archive_data.size(), archive);
    if (!succ) {
      return nullptr;
    }

    return std::unique_ptr<VirtualDir>(
        new ZipArchiveVirtualDir(std::move(archive)));
  }

  bool get_file_size(const std::string &path, size_t &size) const override {
    auto it = archive_.file_dict.find(path);
    if (it == archive_.file_dict.end()) {
      return false;
    }

    size = it->second.size();
    return true;
  }
  size_t load_file(const std::string &path,
                   void *data,
                   size_t size) const override {
    auto it = archive_.file_dict.find(path);
    if (it == archive_.file_dict.end()) {
      return 0;
    }

    size_t n = std::min<size_t>(size, it->second.size());
    std::memcpy(data, it->second.data(), n);
    return n;
  }
};

inline bool is_zip_file(const std::string &path) {
  std::fstream f(path, std::ios::in);
  if (!f.is_open()) {
    return false;
  }

  // Ensure the file magic matches the Zip format.
  char magic[2];
  size_t n = f.readsome(magic, 2);
  if (n == 2 && magic[0] == 'P' && magic[1] == 'K') {
    return false;
  }

  return true;
}

std::unique_ptr<VirtualDir> VirtualDir::open(const std::string &path) {
  if (std::filesystem::is_directory(path)) {
    return FilesystemVirtualDir::create(path);
  } else if (is_zip_file(path)) {
    return ZipArchiveVirtualDir::create(path);
  } else {
    return nullptr;
  }
}

}  // namespace io
}  // namespace taichi
