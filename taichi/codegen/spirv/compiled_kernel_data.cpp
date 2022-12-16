#include "taichi/codegen/compiled_kernel_data.h"

#include <numeric>

#include "taichi/codegen/spirv/cxompiled_kernel_data.h"
#include "taichi/common/serialization.h"

namespace taichi::lang {
namespace spirv {

CompiledKernelData::CompiledKernelData(InternalData data)
    : data_(std::move(data)) {
}

std::size_t CompiledKernelData::size() const {
  return sizeof(std::uint32_t) *
         std::accumulate(
             data_.spirv_src.begin(), data_.spirv_src.end(), (std::size_t)0,
             [](std::size_t val, const std::vector<std::uint32_t> &c) {
               return val + c.size();
             });
}

CompiledKernelData::Err CompiledKernelData::load(std::istream &is) {
  // FIXME: Not safe & High overhead & ...
  auto ok = read_from_binary_stream(data_, is);
  return ok ? Err::kNoError : Err::kFailed;
}

CompiledKernelData::Err CompiledKernelData::dump(std::ostream &os) const {
  // FIXME: Not safe & High overhead & ...
  write_to_binary_stream(data_, os);
  return os ? Err::kNoError : Err::kFailed;
}

std::unique_ptr<lang::CompiledKernelData> CompiledKernelData::clone() const {
  return std::make_unique<CompiledKernelData>(data_);
}

}  // namespace spirv
}  // namespace taichi::lang
