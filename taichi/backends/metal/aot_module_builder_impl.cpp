#include "taichi/backends/metal/aot_module_builder_impl.h"

#include <fstream>

#include "taichi/backends/metal/codegen_metal.h"

#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ < 8)
// https://stackoverflow.com/a/45867491
// !defined(__clang__) to make sure this is not clang
// https://stackoverflow.com/questions/38499462/how-to-tell-clang-to-stop-pretending-to-be-other-compilers
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
    const CompiledStructs *compiled_structs,
    BufferSize buffer_size_data)
    : compiled_structs_(compiled_structs), buffer_size_data_(buffer_size_data) {
  ti_file_data.sizes = buffer_size_data;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const fs::path dir{output_dir};
  const fs::path bin_path = dir / fmt::format("{}_metadata.tcb", filename);
  write_to_binary_file(ti_file_data, bin_path.string());
  // The txt file is mostly for debugging purpose.
  const fs::path txt_path = dir / fmt::format("{}_metadata.txt", filename);
  TextSerializer ts;
  ts("taichi file data", ti_file_data);
  ts.write_to_file(txt_path.string());

  for (const auto &k : ti_file_data.kernels) {
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
  ti_file_data.kernels.push_back(std::move(compiled));
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
