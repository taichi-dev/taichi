#pragma once

#include <string>
#include <vector>

//#include "taichi/backends/wasm/aot_"
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
  std::vector<std::pair<std::unique_ptr<llvm::Module>, std::string>> modules;
};

}
}
}