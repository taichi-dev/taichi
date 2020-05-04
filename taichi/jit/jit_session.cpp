#include "taichi/jit/jit_session.h"

#include "llvm/IR/DataLayout.h"

TLANG_NAMESPACE_BEGIN

std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Arch arch);
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Arch arch);

std::unique_ptr<JITSession> JITSession::create(Arch arch) {
  if (arch_is_cpu(arch)) {
    return create_llvm_jit_session_cpu(arch);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return create_llvm_jit_session_cuda(arch);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else
    TI_NOT_IMPLEMENTED
}

std::size_t JITSession::get_type_size(llvm::Type *type) {
  return get_data_layout().getTypeAllocSize(type);
}

llvm::DataLayout JITSession::get_data_layout(){TI_NOT_IMPLEMENTED}

TLANG_NAMESPACE_END
