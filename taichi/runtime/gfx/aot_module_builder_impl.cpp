#include "taichi/runtime/gfx/aot_module_builder_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/aot/module_data.h"
#include "taichi/codegen/spirv/spirv_codegen.h"
#include "taichi/runtime/gfx/aot_graph_data.h"

namespace taichi {
namespace lang {
namespace gfx {

namespace {
class AotDataConverter {
 public:
  static aot::ModuleData convert(const TaichiAotData &in) {
    AotDataConverter c{};
    return c.visit(in);
  }

 private:
  explicit AotDataConverter() = default;

  aot::ModuleData visit(const TaichiAotData &in) const {
    aot::ModuleData res{};
    for (const auto &ker : in.kernels) {
      auto val = visit(ker);
      res.kernels[ker.name] = val;
    }
    res.fields = in.fields;
    res.required_caps = in.required_caps;
    res.root_buffer_size = in.root_buffer_size;
    return res;
  }

  aot::CompiledTaichiKernel visit(
      const spirv::TaichiKernelAttributes &in) const {
    aot::CompiledTaichiKernel res{};
    res.tasks.reserve(in.tasks_attribs.size());
    for (const auto &t : in.tasks_attribs) {
      res.tasks.push_back(visit(t));
    }
    res.args_count = in.ctx_attribs.args().size();
    res.rets_count = in.ctx_attribs.rets().size();
    res.args_buffer_size = in.ctx_attribs.args_bytes();
    res.rets_buffer_size = in.ctx_attribs.rets_bytes();
    for (const auto &arg : in.ctx_attribs.args()) {
      if (!arg.is_array) {
        aot::ScalarArg scalar_arg{};
        scalar_arg.dtype_name = PrimitiveType::get(arg.dtype).to_string();
        scalar_arg.offset_in_args_buf = arg.offset_in_mem;
        res.scalar_args[arg.index] = scalar_arg;
      } else {
        aot::ArrayArg arr_arg{};
        arr_arg.dtype_name = PrimitiveType::get(arg.dtype).to_string();
        arr_arg.field_dim = arg.field_dim;
        arr_arg.element_shape = arg.element_shape;
        arr_arg.shape_offset_in_args_buf = arg.index * sizeof(int32_t);
        res.arr_args[arg.index] = arr_arg;
      }
    }
    return res;
  }

  aot::CompiledOffloadedTask visit(const TaskAttributes &in) const {
    aot::CompiledOffloadedTask res{};
    res.type = offloaded_task_type_name(in.task_type);
    res.name = in.name;
    if (in.range_for_attribs && in.range_for_attribs->const_begin &&
        in.range_for_attribs->const_end) {
      res.range_hint = std::to_string(in.range_for_attribs->end -
                                      in.range_for_attribs->begin);
    }
    res.gpu_block_size = in.advisory_num_threads_per_group;
    for (auto &buffer_bind : in.buffer_binds) {
      if (buffer_bind.buffer.type == BufferType::Root) {
        res.buffer_binds.push_back(
            {{aot::BufferType::Root, buffer_bind.buffer.root_id},
             buffer_bind.binding});
      } else if (buffer_bind.buffer.type == BufferType::Rets) {
        res.buffer_binds.push_back(
            {{aot::BufferType::Rets, buffer_bind.buffer.root_id},
             buffer_bind.binding});
      } else if (buffer_bind.buffer.type == BufferType::GlobalTmps) {
        res.buffer_binds.push_back(
            {{aot::BufferType::GlobalTmps, buffer_bind.buffer.root_id},
             buffer_bind.binding});
      } else if (buffer_bind.buffer.type == BufferType::Args) {
        res.buffer_binds.push_back(
            {{aot::BufferType::Args, buffer_bind.buffer.root_id},
             buffer_bind.binding});
      }
    }

    for (auto &texture_bind : in.texture_binds) {
      res.texture_binds.push_back(
          {texture_bind.arg_id, texture_bind.binding, texture_bind.is_storage});
    }
    return res;
  }
};

}  // namespace
AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<CompiledSNodeStructs> &compiled_structs,
    Arch device_api_backend,
    std::unique_ptr<Device> &&target_device)
    : compiled_structs_(compiled_structs),
      device_api_backend_(device_api_backend) {
  aot_target_device_ =
      target_device ? std::move(target_device)
                    : std::make_unique<aot::TargetDevice>(device_api_backend_);
  aot_target_device_->clone_caps(ti_aot_data_.required_caps);
  if (!compiled_structs.empty()) {
    ti_aot_data_.root_buffer_size = compiled_structs[0].root_size;
  }
}

std::string AotModuleBuilderImpl::write_spv_file(
    const std::string &output_dir,
    const TaskAttributes &k,
    const std::vector<uint32_t> &source_code) const {
  const std::string spv_path = fmt::format("{}/{}.spv", output_dir, k.name);
  std::ofstream fs(spv_path, std::ios_base::binary | std::ios::trunc);
  fs.write((char *)source_code.data(), source_code.size() * sizeof(uint32_t));
  fs.close();
  return spv_path;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on Unified Device API backends.");
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  write_to_binary_file(ti_aot_data_, bin_path);

  auto converted = AotDataConverter::convert(ti_aot_data_);
  const auto &spirv_codes = ti_aot_data_.spirv_codes;
  for (int i = 0; i < std::min(ti_aot_data_.kernels.size(), spirv_codes.size());
       ++i) {
    auto &k = ti_aot_data_.kernels[i];
    for (int j = 0; j < std::min(k.tasks_attribs.size(), spirv_codes[i].size());
         ++j) {
      if (!spirv_codes[i][j].empty()) {
        std::string spv_path =
            write_spv_file(output_dir, k.tasks_attribs[j], spirv_codes[i][j]);
        converted.kernels[k.name].tasks[j].source_path = spv_path;
      }
    }
  }

  const std::string json_path = fmt::format("{}/metadata.json", output_dir);
  converted.dump_json(json_path);

  dump_graph(output_dir);
}

void AotModuleBuilderImpl::mangle_aot_data() {
  // Only for offline cache
  for (auto &kernel : ti_aot_data_.kernels) {
    const auto &prefix = kernel.name;
    for (std::size_t i = 0; i < kernel.tasks_attribs.size(); ++i) {
      kernel.tasks_attribs[i].name = prefix + std::to_string(i);
    }
  }
}

void AotModuleBuilderImpl::merge_with_old_meta_data(const std::string &path) {
  // Only for offline cache
  auto filename = taichi::join_path(path, "metadata.tcb");
  if (taichi::path_exists(filename)) {
    TaichiAotData old_data;
    read_from_binary_file(old_data, filename);
    // Ignore root_buffer_size and fields which aren't needed for offline cache
    ti_aot_data_.kernels.insert(ti_aot_data_.kernels.end(),
                                old_data.kernels.begin(),
                                old_data.kernels.end());
  }
}

std::optional<GfxRuntime::RegisterParams>
AotModuleBuilderImpl::try_get_kernel_register_params(
    const std::string &kernel_name) const {
  const auto &kernels = ti_aot_data_.kernels;
  for (std::size_t i = 0; i < kernels.size(); ++i) {
    if (kernels[i].name == kernel_name) {
      GfxRuntime::RegisterParams result;
      result.kernel_attribs = kernels[i];
      result.task_spirv_source_codes = ti_aot_data_.spirv_codes[i];
      // We only support a single SNodeTree during AOT.
      result.num_snode_trees = 1;
      return result;
    }
  }
  return std::nullopt;
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  spirv::lower(kernel);
  auto compiled =
      run_codegen(kernel, aot_target_device_.get(), compiled_structs_);
  compiled.kernel_attribs.name = identifier;
  ti_aot_data_.kernels.push_back(compiled.kernel_attribs);
  ti_aot_data_.spirv_codes.push_back(compiled.task_spirv_source_codes);
}

void AotModuleBuilderImpl::add_compiled_kernel(aot::Kernel *kernel) {
  const auto register_params = static_cast<KernelImpl *>(kernel)->params();
  ti_aot_data_.kernels.push_back(register_params.kernel_attribs);
  ti_aot_data_.spirv_codes.push_back(register_params.task_spirv_source_codes);
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  // Note that currently we only support adding dense fields in AOT for all
  // backends. In opengl backend we only error out when a non dense field is
  // added to the aot module, but in metal backend we error out earlier when
  // constructing aot module. Ideally we will unify this behavior but it doesn't
  // matter too much for now.
  TI_ERROR_IF(!all_fields_are_dense_in_container(rep_snode->parent),
              "AOT: only supports dense field");

  const auto &dense_desc =
      compiled_structs_[0].snode_descriptors.at(rep_snode->parent->id);

  aot::CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.is_scalar = is_scalar;
  field_data.dtype = static_cast<int>(dt->cast<PrimitiveType>()->type);
  field_data.dtype_name = dt.to_string();
  field_data.shape = shape;
  field_data.mem_offset_in_parent = dense_desc.mem_offset_in_parent_cell;
  if (!is_scalar) {
    field_data.element_shape = {row_num, column_num};
  }
  ti_aot_data_.fields.push_back(field_data);
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  spirv::lower(kernel);
  auto compiled =
      run_codegen(kernel, aot_target_device_.get(), compiled_structs_);

  compiled.kernel_attribs.name = identifier + "|" + key;
  ti_aot_data_.kernels.push_back(compiled.kernel_attribs);
  ti_aot_data_.spirv_codes.push_back(compiled.task_spirv_source_codes);
}

}  // namespace gfx
}  // namespace lang
}  // namespace taichi
