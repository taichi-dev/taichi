#include "taichi/backends/opengl/aot_module_builder_impl.h"

namespace taichi {
namespace lang {
namespace opengl {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    StructCompiledResult &compiled_structs,
    OpenGlRuntime &runtime)
    : compiled_structs_(compiled_structs), runtime_(runtime) {
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const std::string bin_path =
      fmt::format("{}/{}_metadata.tcb", output_dir, filename);
  write_to_binary_file(aot_kernels_, bin_path);
  // The txt file is mostly for debugging purpose.
  const std::string txt_path =
      fmt::format("{}/{}_metadata.txt", output_dir, filename);
  TextSerializer ts;
  ts("taichi aot data", aot_kernels_);
  ts.write_to_file(txt_path);
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_, &runtime_);
  auto compiled = codegen.compile(*kernel);
  aot_kernels_.push_back({compiled, identifier});
}

void AotModuleBuilderImpl::add_per_backend_field(const std::string &identifier,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_, &runtime_);
  auto compiled = codegen.compile(*kernel);

  for (auto &k : aot_kernel_tmpls_) {
    if (k.identifier == identifier) {
      k.program.insert(std::make_pair(key, compiled));
      return;
    }
  }

  CompiledKernelTmpl tmpldata;
  tmpldata.identifier = identifier;
  tmpldata.program.insert(std::make_pair(key, compiled));

  aot_kernel_tmpls_.push_back(std::move(tmpldata));
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi