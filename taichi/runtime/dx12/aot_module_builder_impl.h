#pragma once

#include "taichi/aot/module_builder.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/llvm/llvm_aot_module_builder.h"
#include "taichi/aot/module_data.h"

namespace taichi::lang {
namespace directx12 {

struct ModuleDataDX12 : public aot::ModuleData {
  std::unordered_map<std::string, std::vector<std::vector<uint8_t>>> dxil_codes;
};

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(const CompileConfig &config,
                                LlvmProgramImpl *prog,
                                TaichiLLVMContext &tlctx);

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

  const CompileConfig &config_;
  LlvmProgramImpl *prog;
  ModuleDataDX12 module_data;
  TaichiLLVMContext &tlctx_;
};

}  // namespace directx12
}  // namespace taichi::lang
