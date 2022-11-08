#include "gtest/gtest.h"
#include "taichi/common/version.h"
#include "taichi/util/offline_cache.h"

#ifdef TI_WITH_LLVM
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#endif  // TI_WITH_LLVM

namespace taichi::lang {

namespace {

namespace oc = offline_cache;

inline void gen_old_version(oc::Version &ver) {
  auto &[major, minor, patch] = ver;
  major = std::max(TI_VERSION_MAJOR - 1, 0);
  minor = std::max(TI_VERSION_MINOR - 1, 0);
  patch = std::max(TI_VERSION_PATCH - 1, 0);
}

template <typename MetadataType>
MetadataType gen_metadata(const oc::Version &ver) {
  MetadataType result;
  result.size = 1024;
  std::copy(std::begin(ver), std::end(ver), std::begin(result.version));
  result.kernels["1"] = {};
  result.kernels["2"] = {};
  return result;
}

template <typename MetadataType>
MetadataType gen_old_metadata() {
  oc::Version old_ver{};
  gen_old_version(old_ver);
  return gen_metadata<MetadataType>(old_ver);
}

template <typename MetadataType>
MetadataType gen_correct_metadata() {
  oc::Version ver{TI_VERSION_MAJOR, TI_VERSION_MINOR, TI_VERSION_PATCH};
  return gen_metadata<MetadataType>(ver);
}

template <typename MetadataType>
void load_metadata_test() {
  std::string fake_file = fmt::format("{}.tcb", std::tmpnam(nullptr));
  std::string old_file = fmt::format("{}.tcb", std::tmpnam(nullptr));
  std::string corrupted_file = fmt::format("{}.tcb", std::tmpnam(nullptr));
  std::string true_file = fmt::format("{}.tcb", std::tmpnam(nullptr));

  // Generate metadata & Save as file
  write_to_binary_file(gen_correct_metadata<MetadataType>(), true_file);
  // Generate old metadata & Save as file
  write_to_binary_file(gen_old_metadata<MetadataType>(), old_file);
  // Generate corrupted metadata file
  write_to_binary_file(gen_correct_metadata<MetadataType>(), corrupted_file);
  std::ofstream(corrupted_file, std::ios::app | std::ios::binary)
      << "I-AM-BAD-BYTES" << std::flush;

  using Error = oc::LoadMetadataError;
  Error error = Error::kNoError;

  // Load a non-existing metadata file
  {
    MetadataType data;
    error = oc::load_metadata_with_checking(data, fake_file);
    EXPECT_EQ(error, Error::kFileNotFound);
  }
  // Load a old metadata file
  {
    MetadataType data;
    error = oc::load_metadata_with_checking(data, old_file);
    EXPECT_EQ(error, Error::kVersionNotMatched);
  }
  // Load a corrupted metadata file
  {
    MetadataType data;
    error = oc::load_metadata_with_checking(data, corrupted_file);
    EXPECT_EQ(error, Error::kCorrupted);
  }
  // Load a correct metadata file
  {
    MetadataType data;
    error = oc::load_metadata_with_checking(data, true_file);
    auto [major, minor, patch] = data.version;
    EXPECT_EQ(error, Error::kNoError);
    EXPECT_EQ(major, TI_VERSION_MAJOR);
    EXPECT_EQ(minor, TI_VERSION_MINOR);
    EXPECT_EQ(patch, TI_VERSION_PATCH);
    EXPECT_EQ(data.size, 1024);
    EXPECT_TRUE(data.kernels.count("1"));
    EXPECT_TRUE(data.kernels.count("2"));
  }

  taichi::remove(old_file);
  taichi::remove(corrupted_file);
  taichi::remove(true_file);
}

}  // namespace

// FIXME: (penguinliong) This structure has a same prototype as the actual types
// including `OfflineCacheKernelMetadata`s. It's currently used only for the
// tests and should probably be removed in the future.
struct KernelMetadataBase {
  std::string kernel_key;
  std::size_t size{0};          // byte
  std::time_t created_at{0};    // sec
  std::time_t last_used_at{0};  // sec

  TI_IO_DEF(kernel_key, size, created_at, last_used_at);
};

TEST(OfflineCache, LoadMetadata) {
#ifdef TI_WITH_LLVM
  load_metadata_test<LlvmOfflineCache>();
#endif  // TI_WITH_LLVM
  load_metadata_test<oc::Metadata<KernelMetadataBase>>();
}

}  // namespace taichi::lang
