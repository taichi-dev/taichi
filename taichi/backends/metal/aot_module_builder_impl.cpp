#include "taichi/backends/metal/aot_module_builder_impl.h"

#include <fstream>

#include "taichi/backends/metal/codegen_metal.h"

#if defined(__GNUC__) && (__GNUC__ < 8)
// https://stackoverflow.com/a/45867491
#include <experimental/filesystem>
namespace fs = ::std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = ::std::filesystem;
#endif

namespace taichi {
namespace lang {
namespace metal {

AotModuleBuilderImpl::AotModuleBuilderImpl(
    const CompiledStructs *compiled_structs)
    : compiled_structs_(compiled_structs) {
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const fs::path dir{output_dir};
  const fs::path bin_path = dir / fmt::format("{}_metadata.tcb", filename);
  write_to_binary_file(kernels_, bin_path.string());
  // The txt file is mostly for debugging purpose.
  const fs::path txt_path = dir / fmt::format("{}_metadata.txt", filename);
  TextSerializer ts;
  ts("kernels", kernels_);
  ts.write_to_file(txt_path.string());

  for (const auto &k : kernels_) {
    const fs::path mtl_path =
        dir / fmt::format("{}_{}.metal", filename, k.kernel_name);
    std::ofstream fs{mtl_path.string()};
    fs << k.source_code;
    fs.close();
  }
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto compiled =
      run_codegen(compiled_structs_, kernel, &strtab_, /*offloaded=*/nullptr);
  compiled.kernel_name = identifier;
  kernels_.push_back(std::move(compiled));
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
