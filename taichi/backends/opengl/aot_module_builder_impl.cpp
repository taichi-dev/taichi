#include "taichi/backends/opengl/aot_module_builder_impl.h"
#include "taichi/backends/opengl/opengl_utils.h"

#if !defined(TI_PLATFORM_WINDOWS)
#include <stdio.h>
#endif

namespace taichi {
namespace lang {
namespace opengl {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    StructCompiledResult &compiled_structs,
    bool allow_nv_shader_extension)
    : compiled_structs_(compiled_structs),
      allow_nv_shader_extension_(allow_nv_shader_extension) {
  aot_data_.root_buffer_size = compiled_structs_.root_size;
}

namespace {
void write_glsl_file(const std::string &output_dir, CompiledKernel &k) {
  const std::string glsl_path =
      fmt::format("{}/{}.glsl", output_dir, k.kernel_name);
  std::ofstream fs{glsl_path};
  fs << k.kernel_src;
  k.kernel_src = glsl_path;
  fs.close();
}

}  // namespace

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on opengl backend.");
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  write_to_binary_file(aot_data_, bin_path);
  // Json format doesn't support multiple line strings.
  AotData new_aot_data = aot_data_;
  for (auto &k : new_aot_data.kernels) {
    for (auto &ki : k.program.kernels) {
      write_glsl_file(output_dir, ki);
    }
  }
  for (auto &k : new_aot_data.kernel_tmpls) {
    for (auto &ki : k.program) {
      for (auto &kij : ki.second.kernels) {
        write_glsl_file(output_dir, kij);
      }
    }
  }

  const std::string txt_path = fmt::format("{}/metadata.json", output_dir);
  TextSerializer ts;
  ts.serialize_to_json("aot_data", new_aot_data);
  ts.write_to_file(txt_path);
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_,
                                allow_nv_shader_extension_);
  auto compiled = codegen.compile(*kernel);
  aot_data_.kernels.push_back({compiled, identifier});
}

size_t AotModuleBuilderImpl::get_snode_base_address(const SNode *snode) {
  if (snode->type == SNodeType::root)
    return 0;
  int chid = find_children_id(snode);
  const auto &parent_meta =
      compiled_structs_.snode_map.at(snode->parent->node_type_name);
  auto choff = parent_meta.children_offsets[chid];
  return choff + get_snode_base_address(snode->parent);
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  uint32_t gl_dtype_enum = to_gl_dtype_enum(dt);

  // Note that currently we only support adding dense fields in AOT for all
  // backends. In opengl backend we only error out when a non dense field is
  // added to the aot module, but in metal backend we error out earlier when
  // constructing aot module. Ideally we will unify this behavior but it doesn't
  // matter too much for now.
  TI_ERROR_IF(!all_fields_are_dense_in_container(rep_snode->parent),
              "AOT: only supports dense field");
  aot_data_.fields.push_back({identifier, gl_dtype_enum, dt.to_string(),
                              get_snode_base_address(rep_snode), shape,
                              is_scalar, row_num, column_num});
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_,
                                allow_nv_shader_extension_);
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
