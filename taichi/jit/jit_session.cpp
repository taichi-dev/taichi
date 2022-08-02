#include "taichi/jit/jit_session.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/DataLayout.h"
#endif

TLANG_NAMESPACE_BEGIN

#ifdef TI_WITH_LLVM
std::unique_ptr<JITSession> create_llvm_jit_session_cpu(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch);

std::unique_ptr<JITSession> create_llvm_jit_session_cuda(
    TaichiLLVMContext *tlctx,
    CompileConfig *config,
    Arch arch);
#endif

JITSession::JITSession(TaichiLLVMContext *tlctx, CompileConfig *config)
    : tlctx_(tlctx), config_(config) {
}

std::unique_ptr<JITSession> JITSession::create(TaichiLLVMContext *tlctx,
                                               CompileConfig *config,
                                               Arch arch) {
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    return create_llvm_jit_session_cpu(tlctx, config, arch);
  } else if (arch == Arch::dx12) {
    return create_llvm_jit_session_cpu(tlctx, config, Arch::x64);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return create_llvm_jit_session_cuda(tlctx, config, arch);
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
