#pragma once
#include "taichi/common/core.h"

namespace taichi {
namespace io {

// A universal filesystem interface for read-only access.
struct TI_DLL_EXPORT VirtualDir {
  virtual ~VirtualDir() {
  }

  // Open a virtual directory based on what `path` points to. Zip files and
  // filesystem directories are supported.
  static std::unique_ptr<VirtualDir> open(const std::string &path);
  static std::unique_ptr<VirtualDir> from_zip(const void *data, size_t size);
  static std::unique_ptr<VirtualDir> from_fs_dir(const std::string &base_dir);

  // Get the `size` of the file at `path` in the virtual directory. Returns
  // false when the file doesn't exist.
  virtual bool get_file_size(const std::string &path, size_t &size) const = 0;
  // Load the first `size` bytes from the file at `path` in the virtual
  // directory. Returns the number of bytes read. Returns 0 if the file doesn't
  // exist.
  virtual size_t load_file(const std::string &path,
                           void *data,
                           size_t size) const = 0;

  template <typename T>
  bool load_file(const std::string &path, std::vector<T> &data) const {
    size_t size = 0;

    if (!get_file_size(path, size)) {
      return false;
    }
    if (size % sizeof(T) != 0) {
      return false;
    }
    data.resize(size / sizeof(T));
    if (load_file(path, data.data(), size) != size) {
      return false;
    }
    return true;
  }
};

}  // namespace io
}  // namespace taichi
