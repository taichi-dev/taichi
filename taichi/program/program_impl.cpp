#include "program_impl.h"

namespace taichi::lang {

ProgramImpl::ProgramImpl(CompileConfig &config_) : config(&config_) {
}

void ProgramImpl::compile_snode_tree_types(SNodeTree *tree) {
  // FIXME: Eventually all the backends should implement this
  TI_NOT_IMPLEMENTED;
}

FunctionType ProgramImpl::compile(const CompileConfig &compile_config,
                                  Kernel *kernel) {
  // NOTE: Temporary implementation (blocked by cc backend)
  // TODO(PGZXB): Final solution: compile -> load_or_compile + launch_kernel
  auto &mgr = get_kernel_compilation_manager();
  const auto &compiled =
      mgr.load_or_compile(compile_config, get_device_caps(), *kernel);
  auto &launcher = get_kernel_launcher();
  return [&launcher, &compiled](LaunchContextBuilder &ctx_builder) {
    launcher.launch_kernel(compiled, ctx_builder);
  };
}

void ProgramImpl::dump_cache_data_to_disk() {
  auto &mgr = get_kernel_compilation_manager();
  mgr.clean_offline_cache(offline_cache::string_to_clean_cache_policy(
                              config->offline_cache_cleaning_policy),
                          config->offline_cache_max_size_of_files,
                          config->offline_cache_cleaning_factor);
  mgr.dump();
}

KernelCompilationManager &ProgramImpl::get_kernel_compilation_manager() {
  if (kernel_com_mgr_) {
    return *kernel_com_mgr_;
  }
  KernelCompilationManager::Config cfg;
  cfg.offline_cache_path = config->offline_cache_file_path;
  cfg.kernel_compiler = make_kernel_compiler();
  kernel_com_mgr_ = std::make_unique<KernelCompilationManager>(std::move(cfg));
  return *kernel_com_mgr_;
}

KernelLauncher &ProgramImpl::get_kernel_launcher() {
  if (kernel_launcher_) {
    return *kernel_launcher_;
  }
  return *(kernel_launcher_ = make_kernel_launcher());
}

}  // namespace taichi::lang
