#include "taichi/backends/vulkan/aot_module_builder_impl.h"

#include <fstream>
#include <type_traits>

#include "taichi/codegen/spirv/spirv_codegen.h"

namespace taichi {
namespace lang {
namespace vulkan {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    const std::vector<CompiledSNodeStructs> &compiled_structs)
    : compiled_structs_(compiled_structs) {
  aot_target_device_ = std::make_unique<AotTargetDevice>(Arch::vulkan);
}

void AotModuleBuilderImpl::write_spv_file(
    const std::string &output_dir,
    const TaskAttributes &k,
    const std::vector<uint32_t> &source_code) const {
  const std::string spv_path = fmt::format("{}/{}.spv", output_dir, k.name);
  std::ofstream fs(spv_path, std::ios_base::binary | std::ios::trunc);
  fs.write((char *)source_code.data(), source_code.size() * sizeof(uint32_t));
  fs.close();
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_WARN_IF(!filename.empty(),
             "Filename prefix is ignored on vulkan backend.");
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  write_to_binary_file(ti_aot_data_, bin_path);
  // Json format doesn't support multiple line strings.
  for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
    auto k = ti_aot_data_.kernels[i];
    for (int j = 0; j < k.tasks_attribs.size(); ++j) {
      write_spv_file(output_dir, k.tasks_attribs[j],
                     ti_aot_data_.spirv_codes[i][j]);
    }
  }

  const std::string txt_path = fmt::format("{}/metadata.json", output_dir);
  TextSerializer ts;
  ts.serialize_to_json("aot_data", ti_aot_data_);
  ts.write_to_file(txt_path);
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  spirv::lower(kernel);
  auto compiled =
      run_codegen(kernel, aot_target_device_.get(), compiled_structs_);
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
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
}

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
