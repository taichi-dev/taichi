#include "taichi/backends/wasm/aot_module_builder_impl.h"

#include <fstream>

#include "taichi/system/std_filesystem.h"
#include "taichi/util/file_sequence_writer.h"

namespace taichi {
namespace lang {
namespace wasm {

AotModuleBuilderImpl::AotModuleBuilderImpl() : module_(nullptr) {
  TI_AUTO_PROF
}

void AotModuleBuilderImpl::eliminate_unused_functions() const {
  TaichiLLVMContext::eliminate_unused_functions(
      module_.get(), [&](std::string func_name) {
        for (auto &name : name_list_) {
          if (name == func_name)
            return true;
        }
        return false;
      });
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  const stdfs::path dir{output_dir};
  const stdfs::path bin_path = dir / fmt::format("{}.ll", filename);

  eliminate_unused_functions();
  FileSequenceWriter writer(bin_path.string(), "optimized LLVM IR (WASM)");
  writer.write(module_.get());
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto module_info = CodeGenWASM(kernel, nullptr).modulegen(std::move(module_));
  module_ = std::move(module_info->module);

  for (auto &name : module_info->name_list)
    name_list_.push_back(name);
}

}  // namespace wasm
}  // namespace lang
}  // namespace taichi
