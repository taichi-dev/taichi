#pragma once

#include <string>
#include <vector>

#include "taichi/program/aot_module_builder.h"
#include "taichi/program/kernel.h"

#include "taichi/backends/wasm/codegen_wasm.h"

namespace taichi {
namespace lang {
namespace wasm {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl();

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

 private:
  void eliminate_unused_functions() const;
  std::unique_ptr<llvm::Module> module_{nullptr};
  std::vector<std::string> name_list_;
};

}  // namespace wasm
}  // namespace lang
}  // namespace taichi
