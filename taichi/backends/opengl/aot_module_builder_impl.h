#pragma once

#include <string>
#include <vector>

#include "taichi/program/aot_module_builder.h"
#include "taichi/backends/opengl/aot_data.h"

namespace taichi {
namespace lang {
namespace opengl {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(StructCompiledResult &compiled_structs);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;
  void add_per_backend_field(const std::string &identifier,
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
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
