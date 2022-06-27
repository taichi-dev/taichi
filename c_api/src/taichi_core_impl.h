#pragma once
#include "taichi/taichi_core.h"
#include "taichi/aot/module_loader.h"
#include "taichi/backends/device.h"
#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

class Runtime;
class Context;
class AotModule;

class Runtime {
 protected:
  // 32 is a magic number in `taichi/inc/constants.h`.
  std::array<uint64_t, 32> host_result_buffer_;

  Runtime(taichi::Arch arch);

 public:
  const taichi::Arch arch;
  taichi::lang::RuntimeContext runtime_context_;

  virtual ~Runtime();

  virtual taichi::lang::Device &get() = 0;

  virtual TiAotModule load_aot_module(const char *module_path) = 0;
  virtual void submit() = 0;
  virtual void wait() = 0;

  class VulkanRuntime *as_vk();
};

class AotModule {
  Runtime *runtime_;
  std::unique_ptr<taichi::lang::aot::Module> aot_module_;
  std::unordered_map<std::string,
                     std::unique_ptr<taichi::lang::aot::CompiledGraph>>
      loaded_cgraphs_;

 public:
  AotModule(Runtime &runtime,
            std::unique_ptr<taichi::lang::aot::Module> &&aot_module);

  taichi::lang::aot::CompiledGraph &get_cgraph(const std::string &name);
  taichi::lang::aot::Module &get();
  Runtime &runtime();
};

namespace {

taichi::lang::DeviceAllocation devmem2devalloc(Runtime &runtime,
                                               TiMemory devmem) {
  return taichi::lang::DeviceAllocation{
      &runtime.get(), (taichi::lang::DeviceAllocationId)((size_t)devmem - 1)};
}
TiMemory devalloc2devmem(const taichi::lang::DeviceAllocation &devalloc) {
  return (TiMemory)((size_t)devalloc.alloc_id + 1);
}

}  // namespace
