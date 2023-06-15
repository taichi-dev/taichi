#include "taichi/runtime/dx12/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/dx12/codegen_dx12.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"

namespace taichi::lang {
namespace directx12 {

AotModuleBuilderImpl::AotModuleBuilderImpl(const CompileConfig &config,
                                           LlvmProgramImpl *prog,
                                           TaichiLLVMContext &tlctx)
    : config_(config), prog(prog), tlctx_(tlctx) {
  // FIXME: set correct root buffer size.
  module_data.root_buffer_size = 1;
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto &dxil_codes = module_data.dxil_codes[identifier];
  auto &compiled_kernel = module_data.kernels[identifier];

  KernelCodeGenDX12 cgen(config_, kernel, kernel->ir.get(), tlctx_);
  auto compiled_data = cgen.compile();
  for (auto &dxil : compiled_data.task_dxil_source_codes) {
    dxil_codes.emplace_back(dxil);
  }
  // FIXME: set compiled kernel.
  compiled_kernel.tasks = compiled_data.tasks;
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  // FIXME: support sparse fields.
  TI_ERROR_IF(!all_fields_are_dense_in_container(rep_snode->parent),
              "AOT: D12 supports only dense fields for now");

  const auto &field = prog->get_cached_field(rep_snode->get_snode_tree_id());

  // const auto &dense_desc =
  //     compiled_structs_[0].snode_descriptors.at(rep_snode->parent->id);

  aot::CompiledFieldData field_data;
  field_data.field_name = identifier;
  field_data.is_scalar = is_scalar;
  field_data.dtype = static_cast<int>(dt->cast<PrimitiveType>()->type);
  field_data.dtype_name = dt.to_string();
  field_data.shape = shape;
  // FIXME: calc mem_offset_in_parent for llvm path.
  field_data.mem_offset_in_parent = field.snode_metas[0].chunk_size;
  // dense_desc.mem_offset_in_parent_cell;
  if (!is_scalar) {
    field_data.element_shape = {row_num, column_num};
  }

  module_data.fields.emplace_back(field_data);
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  // FIXME: share code with add_per_backend.
  auto tmpl_identifier = identifier + "|" + key;

  auto &dxil_codes = module_data.dxil_codes[tmpl_identifier];
  auto &compiled_kernel = module_data.kernels[tmpl_identifier];

  KernelCodeGenDX12 cgen(config_, kernel, kernel->ir.get(), tlctx_);
  auto compiled_data = cgen.compile();
  for (auto &dxil : compiled_data.task_dxil_source_codes) {
    dxil_codes.emplace_back(dxil);
  }
  // set compiled kernel.
}

std::string write_dxil_container(const std::string &output_dir,
                                 const std::string &name,
                                 const std::vector<uint8_t> &source_code) {
  const std::string path = fmt::format("{}/{}.dxc", output_dir, name);
  std::ofstream fs(path, std::ios_base::binary | std::ios::trunc);
  fs.write((char *)source_code.data(), source_code.size() * sizeof(uint8_t));
  fs.close();
  return path;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on Unified Device API backends.");
  const std::string bin_path = fmt::format("{}/metadata_dx12.tcb", output_dir);
  write_to_binary_file(module_data, bin_path);
  // Copy module_data to update task.source_path.
  auto tmp_module_data = module_data;
  for (auto &[name, compiled_kernel] : tmp_module_data.kernels) {
    auto it = tmp_module_data.dxil_codes.find(name);
    TI_ASSERT(it != tmp_module_data.dxil_codes.end());
    auto &dxil_codes = it->second;
    auto &tasks = compiled_kernel.tasks;
    TI_ASSERT(dxil_codes.size() == tasks.size());
    for (int i = 0; i < tasks.size(); ++i) {
      auto &dxil_code = dxil_codes[i];
      auto &task = tasks[i];
      std::string dxil_path =
          write_dxil_container(output_dir, task.name, dxil_code);
      task.source_path = dxil_path;
    }
  }

  const std::string json_path =
      fmt::format("{}/metadata_dx12.json", output_dir);
  tmp_module_data.dump_json(json_path);

  // FIXME: dump graph to different file.
  // dump_graph(output_dir);
}

}  // namespace directx12
}  // namespace taichi::lang
