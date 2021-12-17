#pragma once

#include <string>
#include <vector>

#include "taichi/backends/vulkan/aot_utils.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/program/aot_module_builder.h"

namespace taichi {
namespace lang {
namespace vulkan {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(
      const std::vector<CompiledSNodeStructs> &compiled_structs);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

  void load(const std::string &output_dir) override;

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

  void write_spv_file(const std::string &output_dir,
                      const TaskAttributes &k,
                      const std::vector<uint32_t> &source_code) const;

  std::vector<uint32_t> read_spv_file(const std::string &output_dir,
                                      const TaskAttributes &k);

  uint32_t to_vk_dtype_enum(DataType dt);

  const std::vector<CompiledSNodeStructs> &compiled_structs_;
  TaichiAotData ti_aot_data_;
  std::unique_ptr<Device> aot_target_device_;
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
