#include "taichi/jit/jit_session.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/DataLayout.h"
#endif

TLANG_NAMESPACE_BEGIN

#ifdef TI_WITH_LLVM
std::unique_ptr<JITSession> create_llvm_jit_session_cpu(
    LlvmProgramImpl *llvm_prog,
    Arch arch);

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    LlvmProgramImpl *llvm_prog,
    Arch arch);
#endif

JITSession::JITSession(LlvmProgramImpl *llvm_prog) : llvm_prog_(llvm_prog) {
}

std::unique_ptr<JITSession> JITSession::create(LlvmProgramImpl *llvm_prog,
                                               Arch arch) {
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    return create_llvm_jit_session_cpu(llvm_prog, arch);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return create_llvm_jit_session_cuda(llvm_prog, arch);
#else
    TI_NOT_IMPLEMENTED
#endif
  }
#else
  TI_ERROR("Llvm disabled");
#endif
  return nullptr;
}

TLANG_NAMESPACE_END
