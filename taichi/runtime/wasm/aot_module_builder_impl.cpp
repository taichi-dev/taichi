#include "taichi/runtime/wasm/aot_module_builder_impl.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"

#include <fstream>

#include "taichi/util/file_sequence_writer.h"
#include "llvm/Linker/Linker.h"

namespace taichi::lang {
namespace wasm {

AotModuleBuilderImpl::AotModuleBuilderImpl(const CompileConfig &compile_config,
                                           TaichiLLVMContext &tlctx)
    : compile_config_(compile_config), module_(nullptr), tlctx_(tlctx) {
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
  std::string bin_path = output_dir + "/" + fmt::format("{}.ll", filename);

  eliminate_unused_functions();
  FileSequenceWriter writer(bin_path, "optimized LLVM IR (WASM)");
  writer.write(module_.get());
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  auto module_info = KernelCodeGenWASM(compile_config_, kernel, tlctx_)
                         .compile_kernel_to_module();
  if (module_) {
    llvm::Linker::linkModules(*module_, std::move(module_info.module),
                              llvm::Linker::OverrideFromSrc);
  } else {
    module_ = std::move(module_info.module);
  }
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
}

}  // namespace wasm
}  // namespace taichi::lang
