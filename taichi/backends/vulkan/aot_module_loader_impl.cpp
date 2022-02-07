#include "taichi/backends/vulkan/aot_module_loader_impl.h"

#include <fstream>
#include <type_traits>

namespace taichi {
namespace lang {
namespace vulkan {

AotModuleLoaderImpl::AotModuleLoaderImpl(const std::string &output_dir) {
  const std::string bin_path = fmt::format("{}/metadata.tcb", output_dir);
  read_from_binary_file(ti_aot_data_, bin_path);
  for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
    auto k = ti_aot_data_.kernels[i];
    std::vector<std::vector<uint32_t>> spirv_sources_codes;
    for (int j = 0; j < k.tasks_attribs.size(); ++j) {
      std::vector<uint32_t> res = read_spv_file(output_dir, k.tasks_attribs[j]);
      spirv_sources_codes.push_back(res);
    }
    ti_aot_data_.spirv_codes.push_back(spirv_sources_codes);
  }
}

std::vector<uint32_t> AotModuleLoaderImpl::read_spv_file(
    const std::string &output_dir,
    const TaskAttributes &k) {
  const std::string spv_path = fmt::format("{}/{}.spv", output_dir, k.name);
  std::vector<uint32_t> source_code;
  std::ifstream fs(spv_path, std::ios_base::binary | std::ios::ate);
  size_t size = fs.tellg();
  fs.seekg(0, std::ios::beg);
  source_code.resize(size / sizeof(uint32_t));
  fs.read((char *)source_code.data(), size);
  fs.close();
  return source_code;
}

bool AotModuleLoaderImpl::get_kernel(const std::string &name,
                                     VkRuntime::RegisterParams &kernel) {
  for (int i = 0; i < ti_aot_data_.kernels.size(); ++i) {
    // Offloaded task names encode more than the name of the function, but for
    // AOT, only use the name of the function which should be the first part of
    // the struct
    if (ti_aot_data_.kernels[i].name.rfind(name, 0) == 0) {
      kernel.kernel_attribs = ti_aot_data_.kernels[i];
      kernel.task_spirv_source_codes = ti_aot_data_.spirv_codes[i];
      return true;
    }
  }

  return false;
}

bool AotModuleLoaderImpl::get_field(const std::string &name,
                                    aot::CompiledFieldData &field) {
  TI_ERROR("AOT: get_field for Vulkan not implemented yet");
  return false;
}

size_t AotModuleLoaderImpl::get_root_size() const {
  return ti_aot_data_.root_buffer_size;
}
}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
