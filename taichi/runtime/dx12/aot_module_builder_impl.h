#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/llvm/llvm_aot_module_builder.h"
#include "taichi/aot/module_data.h"

namespace taichi {
namespace lang {
namespace directx12 {

struct ModuleDataDX12 : public aot::ModuleData {
  std::unordered_map<std::string, std::vector<std::vector<uint8_t>>> dxil_codes;
};

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(LlvmProgramImpl *prog);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

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

  void add_compiled_kernel(aot::Kernel *kernel) override;

  LlvmProgramImpl *prog;
  ModuleDataDX12 module_data;
};

}  // namespace directx12
}  // namespace lang
}  // namespace taichi
