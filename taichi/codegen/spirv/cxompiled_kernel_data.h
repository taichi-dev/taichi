#pragma once

#include <memory>

#include "taichi/codegen/compiled_kernel_data.h"
#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi::lang {
namespace spirv {

class CompiledKernelData : public lang::CompiledKernelData {
 public:
  struct InternalData {
    using TaskCode = std::vector<uint32_t>;
    using TasksCode = std::vector<TaskCode>;

    // meta data
    TaichiKernelAttributes kernel_attribs;
    std::size_t num_snode_trees{0};
    // source code
    TasksCode spirv_src;

    TI_IO_DEF(kernel_attribs, num_snode_trees, spirv_src);
  };

  CompiledKernelData() = default;
  explicit CompiledKernelData(InternalData data);

  CompiledKernelData(const CompiledKernelData &) = delete;
  CompiledKernelData &operator=(const CompiledKernelData &) = delete;
  ~CompiledKernelData() override = default;

  std::size_t size() const override;

  Err load(std::istream &is) override;
  Err dump(std::ostream &os) const override;
  std::unique_ptr<lang::CompiledKernelData> clone() const override;

  const InternalData &get_internal_data() const {
    return data_;
  }

 private:
  InternalData data_;
};

}  // namespace spirv
}  // namespace taichi::lang
