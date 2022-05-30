#pragma once

#include <string>
#include <vector>

#include "taichi/aot/module_builder.h"
#include "taichi/backends/opengl/aot_data.h"

namespace taichi {
namespace lang {
namespace opengl {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(StructCompiledResult &compiled_structs,
                                bool allow_nv_shader_extension);

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
  StructCompiledResult &compiled_structs_;
  AotData aot_data_;
  bool allow_nv_shader_extension_ = false;
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
