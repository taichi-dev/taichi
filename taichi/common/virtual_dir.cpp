#include "taichi/common/virtual_dir.h"
#include "taichi/common/zip.h"

namespace taichi {
namespace io {

struct FilesystemVirtualDir : public VirtualDir {
  std::string base_dir_;

  explicit FilesystemVirtualDir(const std::string &base_dir)
      : base_dir_(base_dir) {
  }

  static std::unique_ptr<VirtualDir> create(const std::string &base_dir) {
    std::string base_dir2;
    if (base_dir.empty()) {
      base_dir2 = "./";
    } else if (base_dir.back() != '/') {
      base_dir2 = base_dir + "/";
    } else {
      base_dir2 = base_dir;
    }

    return std::unique_ptr<VirtualDir>(new FilesystemVirtualDir(base_dir2));
  }

  bool get_file_size(const std::string &path, size_t &size) const override {
    std::fstream f(base_dir_ + path,
                   std::ios::in | std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
      return false;
    }
    size = f.tellg();
    return true;
  }
  size_t load_file(const std::string &path,
                   void *data,
                   size_t size) const override {
    std::fstream f(base_dir_ + path, std::ios::in | std::ios::binary);
    if (!f.is_open()) {
      return false;
    }

    f.read((char *)data, size);
    size_t n = f.gcount();
    return n;
  }
};
struct ZipArchiveVirtualDir : public VirtualDir {
  zip::ZipArchive archive_;

  explicit ZipArchiveVirtualDir(zip::ZipArchive &&archive)
      : archive_(std::move(archive)) {
  }

  static std::unique_ptr<VirtualDir> create(const std::string &archive_path) {
    std::fstream f(archive_path,
                   std::ios::in | std::ios::binary | std::ios::ate);
    std::vector<uint8_t> archive_data(f.tellg());
    f.seekg(std::ios::beg);
    f.read((char *)archive_data.data(), archive_data.size());

    return from_zip(archive_data.data(), archive_data.size());
  }
  static std::unique_ptr<VirtualDir> from_zip(const void *data, size_t size) {
    zip::ZipArchive archive;
    bool succ = zip::ZipArchive::try_from_bytes(data, size, archive);
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
  std::fstream f(path, std::ios::in | std::ios::binary);
  if (!f.is_open()) {
    return false;
  }

  // Ensure the file magic matches the Zip format.
  char magic[2];
  f.read(magic, 2);
  size_t n = f.gcount();
  if (n == 2 && magic[0] == 'P' && magic[1] == 'K') {
    return true;
  }

  return false;
}

std::unique_ptr<VirtualDir> VirtualDir::open(const std::string &path) {
  if (is_zip_file(path)) {
    return ZipArchiveVirtualDir::create(path);
  } else {
    // (penguinliong) I wanted to use `std::filesyste::is_directory`. But it
    // seems `<filesystem>` is only supported in MSVC.
    return FilesystemVirtualDir::create(path);
  }
}

std::unique_ptr<VirtualDir> VirtualDir::from_zip(const void *data,
                                                 size_t size) {
  return ZipArchiveVirtualDir::from_zip(data, size);
}
std::unique_ptr<VirtualDir> VirtualDir::from_fs_dir(
    const std::string &base_dir) {
  return FilesystemVirtualDir::create(base_dir);
}

}  // namespace io
}  // namespace taichi
