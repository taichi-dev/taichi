#include "taichi/backends/wasm/aot_module_builder_impl.h"

#include "taichi/util/file_sequence_writer.h"

#include <fstream>
/*
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
*/

namespace taichi {
namespace lang {
namespace wasm {

AotModuleBuilderImpl::AotModuleBuilderImpl() {
    TI_AUTO_PROF
    std::cout << "Aot init!" << std::endl;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  std::cout << "Aot dump " << output_dir << "||" << filename << std::endl;
  FileSequenceWriter writer(
        "function_{:04d}.ll",
        "optimized LLVM IR (CPU)");
  
  for(auto it=modules.begin(); it < modules.end(); it ++) {
    writer.write(it->first.get());

    std::cout << "Dump " << it->second << std::endl;
  }
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  std::cout << "Aot add " << identifier << std::endl;
  modules.push_back(std::pair<std::unique_ptr<llvm::Module>, std::string>(
      CodeGenWASM(kernel, nullptr).modulegen(), identifier));
}

}
}
}