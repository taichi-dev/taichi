#pragma once
#include "taichi/common/core.h"

namespace taichi {
namespace zip {

// (penguinliong) Currently only supports loading.
struct ZipArchive {
  std::unordered_map<std::string, std::vector<uint8_t>> file_dict;

  ZipArchive() = default;
  ZipArchive(const ZipArchive &) = delete;
  ZipArchive(ZipArchive &&) = default;

  ZipArchive &operator=(const ZipArchive &) = delete;
  ZipArchive &operator=(ZipArchive &&) = default;

  // Parse a serialized Zip archive and extract the uncompressed data keyed by
  // its file name. Returns true if success.
  static bool try_from_bytes(const void *data, size_t size, ZipArchive &ar);
};

}  // namespace zip
}  // namespace taichi
