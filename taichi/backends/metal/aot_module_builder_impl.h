#pragma once

#include <string>
#include <vector>
#include <unordered_set>

#include "taichi/aot/module_builder.h"
#include "taichi/backends/metal/aot_utils.h"
#include "taichi/backends/metal/struct_metal.h"

namespace taichi {
namespace lang {
namespace metal {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(
      const CompiledRuntimeModule *compiled_runtime_module,
      const std::vector<CompiledStructs> &compiled_snode_trees,
      const std::unordered_set<const SNode *> &fields,
      BufferMetaData buffer_meta_data);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
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

 private:
  void write_metal_file(const std::string &dir,
                        const std::string &filename,
                        const CompiledKernelData &k) const;

  const CompiledRuntimeModule *compiled_runtime_module_;
  const std::vector<CompiledStructs> &compiled_snode_trees_;
  const std::unordered_set<const SNode *> fields_;
  PrintStringTable strtab_;
  TaichiAotData ti_aot_data_;
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
