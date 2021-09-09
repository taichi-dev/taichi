#pragma once
#include "taichi/lang_util.h"

#include <vector>

#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/snode_struct_compiler.h"
#include "taichi/backends/vulkan/kernel_utils.h"
#include "taichi/program/compile_config.h"
#include "taichi/struct/snode_tree.h"
#include "taichi/program/snode_expr_utils.h"

namespace taichi {
namespace lang {
namespace vulkan {

class VkRuntime {
 private:
  class Impl;

 public:
  struct Params {
    uint64_t *host_result_buffer = nullptr;
  };

  explicit VkRuntime(const Params &params);
  // To make Pimpl + std::unique_ptr work
  ~VkRuntime();

  class KernelHandle {
   private:
    friend class Impl;
    int id_ = -1;
  };

  struct RegisterParams {
    TaichiKernelAttributes kernel_attribs;
    std::vector<std::vector<uint32_t>> task_spirv_source_codes;
  };

  KernelHandle register_taichi_kernel(RegisterParams params);

  void launch_kernel(KernelHandle handle, Context *host_ctx);

  void materialize_snode_tree(SNodeTree *tree);

  void destroy_snode_tree(SNodeTree *snode_tree);

  void synchronize();

  Device *get_ti_device() const;

  const std::vector<CompiledSNodeStructs> &get_compiled_structs() const;

 private:
  std::unique_ptr<Impl> impl_;
};

bool is_vulkan_api_available();

}  // namespace vulkan
}  // namespace lang
}  // namespace taichi
