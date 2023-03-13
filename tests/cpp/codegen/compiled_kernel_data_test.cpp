#include "gtest/gtest.h"

#include "taichi/common/core.h"
#include "taichi/codegen/compiled_kernel_data.h"

namespace taichi::lang {
namespace {

static constexpr Arch kFakeArch = (Arch)1024;

class FakeCompiledKernelData : public CompiledKernelData {
 public:
  FakeCompiledKernelData() = default;

  FakeCompiledKernelData(std::vector<std::string> func_names,
                         std::string so_bin)
      : compiled_data_{{std::move(func_names)}, std::move(so_bin)} {
  }
  FakeCompiledKernelData(const FakeCompiledKernelData &o)
      : compiled_data_(o.compiled_data_) {
  }

  Arch arch() const override {
    return kFakeArch;
  }

  std::unique_ptr<CompiledKernelData> clone() const override {
    return std::make_unique<FakeCompiledKernelData>(*this);
  }

 protected:
  Err load_impl(const CompiledKernelDataFile &file) override {
    if (file.arch() != kFakeArch) {
      return CompiledKernelData::Err::kArchNotMatched;
    }
    try {
      liong::json::deserialize(liong::json::parse(file.metadata()),
                               compiled_data_.metadata);
    } catch (const liong::json::JsonException &) {
      return CompiledKernelData::Err::kParseMetadataFailed;
    }
    compiled_data_.so_bin = file.src_code();
    return CompiledKernelData::Err::kNoError;
  }

  Err dump_impl(CompiledKernelDataFile &file) const override {
    file.set_arch(kFakeArch);
    try {
      file.set_metadata(
          liong::json::print(liong::json::serialize(compiled_data_.metadata)));
    } catch (const liong::json::JsonException &) {
      return CompiledKernelData::Err::kSerMetadataFailed;
    }
    file.set_src_code(compiled_data_.so_bin);
    return CompiledKernelData::Err::kNoError;
  }

 public:
  struct InternalData {
    struct Metadata {
      std::vector<std::string> func_names;
      TI_IO_DEF(func_names);
    } metadata;
    std::string so_bin;
  } compiled_data_;
};
}  // namespace

TEST(CompiledKernelDataTest, Correct) {
  using Err = CompiledKernelData::Err;
  using FErr = CompiledKernelDataFile::Err;

  std::vector<std::string> func_names = {"offloaded_1", "offloaded_2",
                                         "offloaded_3"};
  std::string so_bin = "I am a so...";

  auto fckd = std::make_unique<FakeCompiledKernelData>(func_names, so_bin);

  std::ostringstream oss;
  EXPECT_EQ(fckd->dump(oss), Err::kNoError);
  auto ser_data = oss.str();

  {
    CompiledKernelDataFile file;
    std::istringstream iss(ser_data);
    EXPECT_EQ(file.load(iss), FErr::kNoError);
    EXPECT_EQ(file.arch(), kFakeArch);
    EXPECT_EQ(file.metadata(),
              liong::json::print(liong::json::serialize(
                  FakeCompiledKernelData::InternalData::Metadata{func_names})));
    EXPECT_EQ(file.src_code(), so_bin);
  }

  {
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::istringstream iss(ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kNoError);
    EXPECT_EQ(fckd->compiled_data_.metadata.func_names, func_names);
    EXPECT_EQ(fckd->compiled_data_.so_bin, so_bin);
  }
}

TEST(CompiledKernelDataTest, Error) {
  using Err = CompiledKernelData::Err;
  using FErr = CompiledKernelDataFile::Err;

  std::vector<std::string> func_names = {"f_1", "f_2", "f_3"};
  std::string so_bin = "I am a so...";
  std::string metadata_j = "{ \"func_names\" : [ \"f_1\", \"f_2\", \"f_3\" ] }";

  CompiledKernelDataFile file;
  file.set_arch(kFakeArch);
  file.set_metadata(metadata_j);
  file.set_src_code(so_bin);

  {  // Correct
    std::ostringstream oss;
    EXPECT_EQ(file.dump(oss), FErr::kNoError);
    auto ser_data = oss.str();
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::istringstream iss(ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kNoError);
    EXPECT_EQ(fckd->compiled_data_.metadata.func_names, func_names);
    EXPECT_EQ(fckd->compiled_data_.so_bin, so_bin);
  }

  {  // Not TIC File
    std::ostringstream oss;
    EXPECT_EQ(file.dump(oss), FErr::kNoError);
    auto ser_data = oss.str();
    ser_data[0] = 'B';  // 'T' -> 'B'
    ser_data[1] = 'A';  // 'I' -> 'A'
    ser_data[2] = 'D';  // 'C' -> 'D'
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::istringstream iss(ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kNotTicFile);
  }

  {  // Corrupted File
    std::ostringstream oss;
    EXPECT_EQ(file.dump(oss), FErr::kNoError);
    auto ser_data = oss.str();
    auto pos =
        ser_data.size() - (CompiledKernelDataFile::kHashSize + so_bin.size());
    ser_data[pos] = 'B';      // 'I' -> 'B'
    ser_data[pos + 1] = '*';  // ' ' -> '*'
    ser_data[pos + 2] = 'A';  // 'a' -> 'A'
    ser_data[pos + 3] = 'D';  // 'm' -> 'D'
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::istringstream iss(ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kCorruptedFile);
  }

  {  // Parse Metadata Failed
    auto bad_metadata_j = "{ \"func_names\" : [ \"f_1\", \"f_2\", \"f_3\" ] ]";
    auto file_copy = file;
    file_copy.set_metadata(bad_metadata_j);
    std::ostringstream oss;
    EXPECT_EQ(file_copy.dump(oss), FErr::kNoError);
    auto ser_data = oss.str();
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::istringstream iss(ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kParseMetadataFailed);
  }

  {  // Arch Not Matched
    auto file_copy = file;
    file_copy.set_arch(Arch::x64);
    std::ostringstream oss;
    EXPECT_EQ(file_copy.dump(oss), FErr::kNoError);
    auto ser_data = oss.str();
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::istringstream iss(ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kArchNotMatched);
  }

  {  // IO Stream Error (is.read(&arch, sizeof(arch)) failed)
    auto fckd = std::make_unique<FakeCompiledKernelData>();
    std::string bad_ser_data = "";
    std::istringstream iss(bad_ser_data);
    EXPECT_EQ(fckd->load(iss), Err::kIOStreamError);
  }
}

}  // namespace taichi::lang
