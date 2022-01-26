#include "taichi/backends/vulkan/aot_module_builder_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/aot/module_data.h"
#include "taichi/codegen/spirv/spirv_codegen.h"

namespace taichi {
namespace lang {
namespace vulkan {

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
      res.scalar_args[arg.index] = visit(arg);
    }
    return res;
  }

  aot::CompiledOffloadedTask visit(const TaskAttributes &in) const {
    aot::CompiledOffloadedTask res{};
    res.type = offloaded_task_type_name(in.task_type);
    res.name = in.name;
    // TODO: update range_hint after ndarray is supported on vulkan.
    if (in.range_for_attribs && in.range_for_attribs->const_begin &&
        in.range_for_attribs->const_end) {
      res.range_hint = std::to_string(in.range_for_attribs->end -
                                      in.range_for_attribs->begin);
    }
    res.gpu_block_size = in.advisory_num_threads_per_group;
    return res;
  }

  aot::ScalarArg visit(
      const spirv::KernelContextAttributes::ArgAttributes &in) const {
    aot::ScalarArg res{};
    res.dtype_name = in.dt.to_string();
    res.offset_in_args_buf = in.offset_in_mem;
    return res;
  }
};

}  // namespace
AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<CompiledSNodeStructs> &compiled_structs)
    : compiled_structs_(compiled_structs) {
  aot_target_device_ = std::make_unique<AotTargetDevice>(Arch::vulkan);
  if (!compiled_structs.empty()) {
    ti_aot_data_.root_buffer_size = compiled_structs[0].root_size;
  }
}

uint32_t AotModuleBuilderImpl::to_vk_dtype_enum(DataType dt) {
  if (dt == PrimitiveType::u64) {
    return 0;
  } else if (dt == PrimitiveType::i64) {
    return 1;
  } else if (dt == PrimitiveType::u32) {
    return 2;
  } else if (dt == PrimitiveType::i32) {
    return 3;
  } else if (dt == PrimitiveType::u16) {
    return 4;
  } else if (dt == PrimitiveType::i16) {
    return 5;
  } else if (dt == PrimitiveType::u8) {
    return 6;
  } else if (dt == PrimitiveType::i8) {
    return 7;
  } else if (dt == PrimitiveType::f64) {
    return 8;
  } else if (dt == PrimitiveType::f32) {
    return 9;
  } else {
    TI_NOT_IMPLEMENTED
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
             "Filename prefix is ignored on vulkan backend.");
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  write_to_binary_file(ti_aot_data_, bin_path);

  auto converted = AotDataConverter::convert(ti_aot_data_);
  for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
    auto &k = ti_aot_data_.kernels[i];
    for (int j = 0; j < k.tasks_attribs.size(); ++j) {
      std::string spv_path = write_spv_file(output_dir, k.tasks_attribs[j],
                                            ti_aot_data_.spirv_codes[i][j]);
      converted.kernels[k.name].tasks[j].source_path = spv_path;
    }
  }

  const std::string json_path = fmt::format("{}/metadata.json", output_dir);
  converted.dump_json(json_path);
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
  field_data.dtype = to_vk_dtype_enum(dt);
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
  TI_ERROR("Templated kernels are not yet supported on vulkan aot.");
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
