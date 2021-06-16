#pragma once

#include <string>
#include <vector>

#include "taichi/backends/metal/taichi_file_util.h"
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/program/aot_module_builder.h"

namespace taichi {
namespace lang {
namespace metal {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(const CompiledStructs *compiled_structs, BufferSize buffer_size_data);

  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;

 private:
  const CompiledStructs *compiled_structs_;
  BufferSize buffer_size_data_;
  PrintStringTable strtab_;
  MetalTiFileData ti_file_data;
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
