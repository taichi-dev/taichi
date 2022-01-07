#include "taichi/backends/opengl/aot_module_builder_impl.h"

#include "taichi/aot/module_data.h"
#include "taichi/backends/opengl/opengl_utils.h"

#if !defined(TI_PLATFORM_WINDOWS)
#include <stdio.h>
#endif

namespace taichi {
namespace lang {
namespace opengl {
namespace {

class AotDataConverter {
 public:
  static aot::ModuleData convert(const opengl::AotData &in) {
    AotDataConverter c{};
    return c.visit(in);
  }

 private:
  explicit AotDataConverter() = default;

  aot::ModuleData visit(const opengl::AotData &in) const {
    aot::ModuleData res{};
    for (const auto &[key, val] : in.kernels) {
      res.kernels[key] = visit(val);
    }
    for (const auto &[key, val] : in.kernel_tmpls) {
      res.kernel_tmpls[key] = visit(val);
    }
    res.fields = in.fields;
    res.root_buffer_size = in.root_buffer_size;
    return res;
  }

  aot::CompiledTaichiKernel visit(
      const opengl::CompiledTaichiKernel &in) const {
    aot::CompiledTaichiKernel res{};
    res.tasks.reserve(in.tasks.size());
    for (const auto &t : in.tasks) {
      res.tasks.push_back(visit(t));
    }
    res.arg_count = in.arg_count;
    res.ret_count = in.ret_count;
    res.args_buf_size = in.args_buf_size;
    res.ret_buf_size = in.ret_buf_size;
    for (const auto &[arg_id, val] : in.scalar_args) {
      res.scalar_args[arg_id] = visit(val);
    }
    for (const auto &[arg_id, val] : in.arr_args) {
      aot::ArrayArg out_arr = visit(val);
      out_arr.bind_index = in.used.arr_arg_to_bind_idx.at(arg_id);
      res.arr_args[arg_id] = out_arr;
    }
    return res;
  }

  aot::CompiledOffloadedTask visit(
      const opengl::CompiledOffloadedTask &in) const {
    aot::CompiledOffloadedTask res{};
    res.type = offloaded_task_type_name(in.type);
    res.name = in.name;
    res.source_path = in.src;
    res.gpu_block_size = in.workgroup_size;
    return res;
  }

  aot::ScalarArg visit(const opengl::ScalarArg &in) const {
    aot::ScalarArg res{};
    res.dtype_name = in.dtype_name;
    res.offset_in_args_buf = in.offset_in_bytes_in_args_buf;
    return res;
  }

  aot::ArrayArg visit(const opengl::CompiledArrayArg &in) const {
    aot::ArrayArg res{};
    res.dtype_name = in.dtype_name;
    res.field_dim = in.field_dim;
    res.element_shape = in.element_shape;
    res.shape_offset_in_args_buf = in.shape_offset_in_bytes_in_args_buf;
    return res;
  }
};

void write_glsl_file(const std::string &output_dir, CompiledOffloadedTask &t) {
  const std::string glsl_path = fmt::format("{}/{}.glsl", output_dir, t.name);
  std::ofstream fs{glsl_path};
  fs << t.src;
  t.src = glsl_path;
  fs.close();
}

}  // namespace

AotModuleBuilderImpl::AotModuleBuilderImpl(
    StructCompiledResult &compiled_structs, bool allow_nv_shader_extension)
    : compiled_structs_(compiled_structs),
      allow_nv_shader_extension_(allow_nv_shader_extension) {
  aot_data_.root_buffer_size = compiled_structs_.root_size;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on opengl backend.");
  // TODO(#3334): Convert |aot_data_| with AotDataConverter
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  write_to_binary_file(aot_data_, bin_path);
  // Json format doesn't support multiple line strings.
  AotData aot_data_copy = aot_data_;
  for (auto &k : aot_data_copy.kernels) {
    for (auto &t : k.second.tasks) {
      write_glsl_file(output_dir, t);
    }
  }
  for (auto &k : aot_data_copy.kernel_tmpls) {
    for (auto &t : k.second.tasks) {
      write_glsl_file(output_dir, t);
    }
  }

  const std::string txt_path = fmt::format("{}/metadata.json", output_dir);
  TextSerializer ts;
  ts.serialize_to_json("aot_data", aot_data_copy);
  ts.write_to_file(txt_path);
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  opengl::OpenglCodeGen codegen(kernel->name, &compiled_structs_,
                                allow_nv_shader_extension_);
  auto compiled = codegen.compile(*kernel);
  aot_data_.kernels.insert(std::make_pair(identifier, std::move(compiled)));
}

size_t AotModuleBuilderImpl::get_snode_base_address(const SNode *snode) {
  if (snode->type == SNodeType::root) return 0;
  int chid = find_children_id(snode);
  const auto &parent_meta =
      compiled_structs_.snode_map.at(snode->parent->node_type_name);
  auto choff = parent_meta.children_offsets[chid];
  return choff + get_snode_base_address(snode->parent);
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar, DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num, int column_num) {
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

  aot_data_.kernel_tmpls.insert(
      std::make_pair(identifier + "|" + key, std::move(compiled)));
}

}  // namespace opengl
}  // namespace lang
}  // namespace taichi
