#include "taichi/backends/opengl/aot_module_builder_impl.h"
#include "glad/glad.h"

namespace taichi {
namespace lang {
namespace opengl {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    StructCompiledResult &compiled_structs)
    : compiled_structs_(compiled_structs) {
  aot_data_.root_buffer_size = compiled_structs_.root_size;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const std::string bin_path =
      fmt::format("{}/{}_metadata.tcb", output_dir, filename);
  write_to_binary_file(aot_data_, bin_path);
  // The txt file is mostly for debugging purpose.
  const std::string txt_path =
      fmt::format("{}/{}_metadata.txt", output_dir, filename);
  TextSerializer ts;
  ts("taichi aot data", aot_data_);
  ts.write_to_file(txt_path);
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_);
  auto compiled = codegen.compile(*kernel);
  aot_data_.kernels.push_back({compiled, identifier});
}

void AotModuleBuilderImpl::add_per_backend_field(const std::string &identifier,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  uint32_t gl_dtype_enum;

  if (dt == PrimitiveType::u64) {
    gl_dtype_enum = GL_UNSIGNED_INT64_ARB;
  } else if (dt == PrimitiveType::i64) {
    gl_dtype_enum = GL_INT64_ARB;
  } else if (dt == PrimitiveType::u32) {
    gl_dtype_enum = GL_UNSIGNED_INT;
  } else if (dt == PrimitiveType::i32) {
    gl_dtype_enum = GL_INT;
  } else if (dt == PrimitiveType::u16) {
    gl_dtype_enum = GL_UNSIGNED_SHORT;
  } else if (dt == PrimitiveType::i16) {
    gl_dtype_enum = GL_SHORT;
  } else if (dt == PrimitiveType::u8) {
    gl_dtype_enum = GL_UNSIGNED_BYTE;
  } else if (dt == PrimitiveType::i8) {
    gl_dtype_enum = GL_BYTE;
  } else if (dt == PrimitiveType::f64) {
    gl_dtype_enum = GL_DOUBLE;
  } else if (dt == PrimitiveType::f32) {
    gl_dtype_enum = GL_FLOAT;
  }

  aot_data_.fields.push_back({identifier, gl_dtype_enum, dt.to_string(), shape,
                              is_scalar, row_num, column_num});
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_);
  auto compiled = codegen.compile(*kernel);

  for (auto &k : aot_data_.kernel_tmpls) {
    if (k.identifier == identifier) {
      k.program.insert(std::make_pair(key, compiled));
      return;
    }
  }

  AotCompiledKernelTmpl tmpldata;
  tmpldata.identifier = identifier;
  tmpldata.program.insert(std::make_pair(key, compiled));

  aot_data_.kernel_tmpls.push_back(std::move(tmpldata));
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
