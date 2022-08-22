#include "taichi/aot/module_loader.h"

#include "taichi/runtime/gfx/aot_module_loader_impl.h"
#include "taichi/runtime/metal/aot_module_loader_impl.h"

namespace taichi {
namespace lang {
namespace aot {
namespace {

std::string make_kernel_key(
    const std::vector<KernelTemplateArg> &template_args) {
  TI_NOT_IMPLEMENTED;
  return "";
}

}  // namespace

Kernel *KernelTemplate::get_kernel(
    const std::vector<KernelTemplateArg> &template_args) {
  const auto key = make_kernel_key(template_args);
  auto itr = loaded_kernels_.find(key);
  if (itr != loaded_kernels_.end()) {
    return itr->second.get();
  }
  auto k = make_new_kernel(template_args);
  auto *kptr = k.get();
  loaded_kernels_[key] = std::move(k);
  return kptr;
}

std::unique_ptr<Module> Module::load(Arch arch, std::any mod_params) {
  if (arch == Arch::vulkan) {
#ifdef TI_WITH_VULKAN
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::opengl) {
#ifdef TI_WITH_OPENGL
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::dx11) {
#ifdef TI_WITH_DX11
    return gfx::make_aot_module(mod_params, arch);
#endif
  } else if (arch == Arch::metal) {
#ifdef TI_WITH_METAL
    return metal::make_aot_module(mod_params);
#endif
  }
  TI_NOT_IMPLEMENTED;
}

Kernel *Module::get_kernel(const std::string &name) {
  auto itr = loaded_kernels_.find(name);
  if (itr != loaded_kernels_.end()) {
    return itr->second.get();
  }
  auto k = make_new_kernel(name);
  auto *kptr = k.get();
  loaded_kernels_[name] = std::move(k);
  return kptr;
}

KernelTemplate *Module::get_kernel_template(const std::string &name) {
  auto itr = loaded_kernel_templates_.find(name);
  if (itr != loaded_kernel_templates_.end()) {
    return itr->second.get();
  }
  auto kt = make_new_kernel_template(name);
  auto *kt_ptr = kt.get();
  loaded_kernel_templates_[name] = std::move(kt);
  return kt_ptr;
}

Field *Module::get_snode_tree(const std::string &name) {
  auto itr = loaded_fields_.find(name);
  if (itr != loaded_fields_.end()) {
    return itr->second.get();
  }
  auto k = make_new_field(name);
  auto *kptr = k.get();
  loaded_fields_[name] = std::move(k);
  return kptr;
}

}  // namespace aot
}  // namespace lang
}  // namespace taichi
