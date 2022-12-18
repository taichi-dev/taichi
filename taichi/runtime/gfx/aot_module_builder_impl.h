#pragma once

#include <string>
#include <vector>

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/gfx/aot_utils.h"
#include "taichi/runtime/gfx/runtime.h"
#include "taichi/cache/kernel_compilation_manager.h"
#include "taichi/codegen/spirv/snode_struct_compiler.h"
#include "taichi/codegen/spirv/kernel_utils.h"

namespace taichi::lang {
namespace gfx {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(
      const std::vector<CompiledSNodeStructs> &compiled_structs,
      KernelCompilationManager *kernel_compilation_manager,
      Arch device_api_backend,
      const CompileConfig &compile_config,
      const DeviceCapabilityConfig &caps);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

  void mangle_aot_data();
  void merge_with_old_meta_data(const std::string &path);
  std::optional<GfxRuntime::RegisterParams> try_get_kernel_register_params(
      const std::string &kernel_name) const;

 private:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

  void add_field_per_backend(const std::string &identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;

  std::string write_spv_file(const std::string &output_dir,
                             const TaskAttributes &k,
                             const std::vector<uint32_t> &source_code) const;

  GfxRuntime::RegisterParams compile_kernel(const Kernel &kernel_def);

  const std::vector<CompiledSNodeStructs> &compiled_structs_;
  KernelCompilationManager *kernel_compilation_manager_{nullptr};
  TaichiAotData ti_aot_data_;

  Arch device_api_backend_;
  const CompileConfig &config_;
  DeviceCapabilityConfig caps_;
};

}  // namespace gfx
}  // namespace taichi::lang
