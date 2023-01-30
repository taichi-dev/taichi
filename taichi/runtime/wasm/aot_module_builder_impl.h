#pragma once

#include <string>
#include <vector>

#include "taichi/aot/module_builder.h"
#include "taichi/program/kernel.h"
#include "taichi/runtime/llvm/llvm_fwd.h"

#include "taichi/codegen/wasm/codegen_wasm.h"

namespace taichi::lang {
namespace wasm {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(const CompileConfig &compile_config);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;
  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;
  void add_field_per_backend(const std::string &Identifier,
                             const SNode *rep_snode,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;

 private:
  void eliminate_unused_functions() const;
  const CompileConfig &compile_config_;
  std::unique_ptr<llvm::Module> module_{nullptr};
  std::vector<std::string> name_list_;
};

}  // namespace wasm
}  // namespace taichi::lang
