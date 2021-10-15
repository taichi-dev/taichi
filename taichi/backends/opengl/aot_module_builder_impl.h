#pragma once

#include <string>
#include <vector>

#include "taichi/program/aot_module_builder.h"
#include "taichi/backends/opengl/opengl_program.h"

namespace taichi {
namespace lang {
namespace opengl {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(StructCompiledResult &compiled_structs,
                                OpenGlRuntime &runtime);

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
  OpenGlRuntime &runtime_;

  struct CompiledKernel {
    CompiledProgram program;
    std::string identifier;
    
    TI_IO_DEF(program, identifier);
  };

  std::vector<CompiledKernel> aot_kernels_;

  struct CompiledKernelTmpl {
    std::unordered_map<std::string, CompiledProgram> program;
    std::string identifier;

    TI_IO_DEF(program, identifier);
  };

  std::vector<CompiledKernelTmpl> aot_kernel_tmpls_;
};

}  // namespace opengl
}  // namespace lang
}  // namespace taichi