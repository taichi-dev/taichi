#include "taichi/codegen/spirv/compiled_kernel_data.h"

#include <numeric>

namespace taichi::lang {

static std::unique_ptr<CompiledKernelData> new_spirv_compiled_kernel_data() {
  return std::make_unique<spirv::CompiledKernelData>();
}

CompiledKernelData::Creator *const CompiledKernelData::spriv_creator =
    new_spirv_compiled_kernel_data;

namespace spirv {

CompiledKernelData::CompiledKernelData(Arch arch, InternalData data)
    : arch_(arch), data_(std::move(data)) {
}

Arch CompiledKernelData::arch() const {
  return arch_;
}

std::size_t CompiledKernelData::size() const {
  return sizeof(std::uint32_t) *
         std::accumulate(
             data_.src.spirv_src.begin(), data_.src.spirv_src.end(),
             (std::size_t)0,
             [](std::size_t val, const std::vector<std::uint32_t> &c) {
               return val + c.size();
             });
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  return std::make_unique<CompiledKernelData>(arch_, data_);
}

CompiledKernelData::Err CompiledKernelData::load_impl(
    const CompiledKernelDataFile &file) {
  arch_ = file.arch();
  if (!arch_uses_spirv(arch_)) {
    return Err::kArchNotMatched;
  }
  try {
    liong::json::deserialize(liong::json::parse(file.metadata()),
                             data_.metadata);
  } catch (const liong::json::JsonException &) {
    return Err::kParseMetadataFailed;
  }
  return str2src(file.src_code(), data_.src);
}

CompiledKernelData::Err CompiledKernelData::dump_impl(
    CompiledKernelDataFile &file) const {
  file.set_arch(arch_);
  try {
    file.set_metadata(
        liong::json::print(liong::json::serialize(data_.metadata)));
  } catch (const liong::json::JsonException &) {
    return Err::kSerMetadataFailed;
  }
  std::string str;
  Err err = src2str(data_.src, str);
  file.set_src_code(std::move(str));
  return err;
}

CompiledKernelData::Err CompiledKernelData::src2str(
    const InternalData::Source &src,
    std::string &result) {
  std::ostringstream oss;
  write_to_binary_stream(src, oss);
  if (oss) {
    result = oss.str();
    return Err::kNoError;
  }
  return Err::kSerSrcCodeFailed;
}

CompiledKernelData::Err CompiledKernelData::str2src(
    const std::string &str,
    InternalData::Source &result) {
  return read_from_binary(result, str.data(), str.size())
             ? Err::kNoError
             : Err::kParseSrcCodeFailed;
}

}  // namespace spirv
}  // namespace taichi::lang
