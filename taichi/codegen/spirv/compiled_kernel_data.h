#pragma once

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
    struct Metadata {
      TaichiKernelAttributes kernel_attribs;
      std::size_t num_snode_trees{0};
      TI_IO_DEF(kernel_attribs, num_snode_trees);
    } metadata;
    // source code
    struct Source {
      TasksCode spirv_src;
      TI_IO_DEF(spirv_src);
    } src;
  };

  CompiledKernelData() = default;
  CompiledKernelData(Arch arch, InternalData data);

  std::size_t size() const override;
  std::unique_ptr<lang::CompiledKernelData> clone() const override;

 protected:
  Err load_impl(const CompiledKernelDataFile &file) override;
  Err dump_impl(CompiledKernelDataFile &file) const override;

 private:
  static Err src2str(const InternalData::Source &src, std::string &result);
  static Err str2src(const std::string &str, InternalData::Source &result);

  Arch arch_;
  InternalData data_;
};

}  // namespace spirv

}  // namespace taichi::lang
