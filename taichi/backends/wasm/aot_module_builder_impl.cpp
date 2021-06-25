#include "taichi/backends/wasm/aot_module_builder_impl.h"

#include "taichi/util/file_sequence_writer.h"

#include <fstream>

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
namespace wasm {

AotModuleBuilderImpl::AotModuleBuilderImpl(): module(nullptr) {
    TI_AUTO_PROF
    name_list = std::make_unique<std::vector<std::string>>();
}

void AotModuleBuilderImpl::eliminate_unused_functions() const {
  TaichiLLVMContext::eliminate_unused_functions(
      module.get(), [&](std::string func_name) {
        for (auto &name : *name_list) {
          if (name == func_name)
            return true;
        }
        return false;
      });
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const fs::path dir{output_dir};
  const fs::path bin_path = dir / fmt::format("{}.ll", filename);

  eliminate_unused_functions();
  FileSequenceWriter writer(
        bin_path.string(),
        "optimized LLVM IR (WASM)");
  writer.write(module.get());
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto info = CodeGenWASMAOT(kernel, nullptr, std::move(module)).modulegen();
  module = std::move(info.first);

  auto name_list = info.second;
  for(auto &name: name_list)
    this->name_list->push_back(name);
}

}
}
}