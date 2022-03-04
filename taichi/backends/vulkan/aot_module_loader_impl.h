#pragma once

#include <string>
#include <vector>

#include "taichi/backends/vulkan/aot_utils.h"
#include "taichi/backends/vulkan/runtime.h"
#include "taichi/codegen/spirv/kernel_utils.h"

#include "taichi/aot/module_loader.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VkRuntime;

struct AotModuleParams {
  std::string module_path;
  VkRuntime *runtime{nullptr};
};

class AotModuleImpl : public aot::Module {
 public:
  explicit AotModuleImpl(const AotModuleParams &params);

  std::unique_ptr<aot::Kernel> get_kernel(const std::string &name) override;

  bool get_field(const std::string &name,
                 aot::CompiledFieldData &field) override;
  size_t get_root_size() const override;

 private:
  bool get_kernel_params_by_name(const std::string &name, VkRuntime::RegisterParams &kernel);
  std::unique_ptr<aot::Kernel> make_new_kernel(
      const std::string &name) override;
  std::vector<uint32_t> read_spv_file(const std::string &output_dir,
                                      const TaskAttributes &k);
  TaichiAotData ti_aot_data_;
  VkRuntime *runtime_{nullptr};
};

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
