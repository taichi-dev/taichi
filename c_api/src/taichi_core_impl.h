#pragma once
#include "taichi/taichi_core.h"
#include "taichi/aot/module_loader.h"
#include "taichi/backends/device.h"
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#define TI_RUNTIME_HOST 1
#include "taichi/program/context.h"
#undef TI_RUNTIME_HOST

class Device;
class Context;
class AotModule;

class Device {
 protected:
  Device(taichi::Arch arch);

 public:
  const taichi::Arch arch;
  virtual ~Device();

  virtual taichi::lang::Device &get() = 0;

  virtual Context *create_context() = 0;

  struct VulkanDevice *as_vk();
};

class Context {
  Device *device_;
  taichi::lang::RuntimeContext runtime_context_;

 protected:
  Context(Device &device);

 public:
  virtual ~Context();

  taichi::lang::RuntimeContext &get();
  Device &device();

  virtual void submit() = 0;
  virtual void wait() = 0;

  struct VulkanContext *as_vk();
};

class AotModule {
  Context *context_;
  std::unique_ptr<taichi::lang::aot::Module> aot_module_;

 public:
  AotModule(Context &context,
            std::unique_ptr<taichi::lang::aot::Module> &&aot_module);

  taichi::lang::aot::Module &get();
  Context &context();
};
