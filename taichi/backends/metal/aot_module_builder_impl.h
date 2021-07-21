#pragma once

#include <string>
#include <vector>

#include "taichi/backends/metal/aot_utils.h"
#include "taichi/backends/metal/struct_metal.h"
#include "taichi/program/aot_module_builder.h"
#include "taichi/system/std_filesystem.h"

namespace taichi {
namespace lang {
namespace metal {

class AotModuleBuilderImpl : public AotModuleBuilder {
 public:
  explicit AotModuleBuilderImpl(const CompiledStructs *compiled_structs,
                                const BufferMetaData &buffer_meta_data);
  void dump(const std::string &output_dir,
            const std::string &filename) const override;

 protected:
  void add_per_backend(const std::string &identifier, Kernel *kernel) override;
  void add_per_backend_field(const std::string &identifier,
                             bool is_scalar,
                             DataType dt,
                             std::vector<int> shape,
                             int row_num,
                             int column_num) override;
  void add_per_backend_tmpl(const std::string &identifier,
                            const std::string &key,
                            Kernel *kernel) override;

 private:
  const CompiledStructs *compiled_structs_;
  BufferMetaData buffer_meta_data_;
  PrintStringTable strtab_;
  TaichiAotData ti_aot_data_;
  void metalgen(const stdfs::path &dir,
                const std::string &filename,
                const CompiledKernelData &k) const;
};

}  // namespace metal
}  // namespace lang
}  // namespace taichi
