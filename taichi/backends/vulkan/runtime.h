#pragma once
#include "taichi/lang_util.h"

#include <vector>

#include "taichi/backends/vulkan/snode_struct_compiler.h"
#include "taichi/backends/vulkan/kernel_utils.h"
#include "taichi/program/compile_config.h"

namespace taichi {
namespace lang {
namespace vulkan {

struct VulkanCapabilities;

class VkRuntime {
 private:
  class Impl;

 public:
  struct Params {
    // CompiledSNodeStructs compiled_snode_structs;
    uint64_t *host_result_buffer = nullptr;
    // int root_id;
    const SNodeDescriptorsMap *snode_descriptors = nullptr;
  };

  explicit VkRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~VkRuntime();

  class KernelHandle {
   private:
    friend class Impl;
    int id_ = -1;
  };

  using SpirvBinary = std::vector<uint32_t>;

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<SpirvBinary> task_spirv_source_codes;
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  void launch_kernel(KernelHandle handle, Context *host_ctx);

  void synchronize();

#ifdef TI_WITH_VULKAN
  const VulkanCapabilities &get_capabilities() const;
#endif

 private:
  std::unique_ptr<Impl> impl_;
};

bool is_vulkan_api_available();

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
