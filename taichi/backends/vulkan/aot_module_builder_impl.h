#pragma once

#include <string>
#include <vector>

#include "taichi/aot/module_builder.h"
#include "taichi/backends/vulkan/aot_utils.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi {
namespace lang {
namespace vulkan {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(
      const std::vector<CompiledSNodeStructs> &compiled_structs);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 private:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;

  std::string write_spv_file(const std::string &output_dir,
                             const TaskAttributes &k,
                             const std::vector<uint32_t> &source_code) const;

  uint32_t to_vk_dtype_enum(DataType dt);

  const std::vector<CompiledSNodeStructs> &compiled_structs_;
  TaichiAotData ti_aot_data_;
  std::unique_ptr<Device> aot_target_device_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
