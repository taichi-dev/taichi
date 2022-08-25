#include "taichi/runtime/dx12/aot_module_builder_impl.h"

#include <algorithm>

#include "taichi/codegen/dx12/codegen_dx12.h"
#include "taichi/runtime/llvm/launch_arg_info.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"

namespace taichi {
namespace lang {
namespace directx12 {

AotModuleBuilderImpl::AotModuleBuilderImpl(LlvmProgramImpl *prog) : prog(prog) {
  // FIXME: set correct root buffer size.
  module_data.root_buffer_size = 1;
}

void AotModuleBuilderImpl::add_per_backend(const std::string &identifier,
                                           Kernel *kernel) {
  TI_NOT_IMPLEMENTED;
}

void AotModuleBuilderImpl::add_compiled_kernel(aot::Kernel *kernel) {
  // FIXME: implement add_compiled_kernel.
  TI_NOT_IMPLEMENTED;
}

void AotModuleBuilderImpl::add_field_per_backend(const std::string &identifier,
                                                 const SNode *rep_snode,
                                                 bool is_scalar,
                                                 DataType dt,
                                                 std::vector<int> shape,
                                                 int row_num,
                                                 int column_num) {
  TI_NOT_IMPLEMENTED;
}

void AotModuleBuilderImpl::add_per_backend_tmpl(const std::string &identifier,
                                                const std::string &key,
                                                Kernel *kernel) {
  TI_NOT_IMPLEMENTED;
}

void AotModuleBuilderImpl::dump(const std::string &output_dir,
                                const std::string &filename) const {
  TI_NOT_IMPLEMENTED;
}

}  // namespace directx12
}  // namespace lang
}  // namespace taichi
