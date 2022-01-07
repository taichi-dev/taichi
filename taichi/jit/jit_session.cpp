#include "taichi/jit/jit_session.h"
#include "taichi/program/program.h"

#ifdef TI_WITH_LLVM
#include "llvm/IR/DataLayout.h"
#endif

TLANG_NAMESPACE_BEGIN

#ifdef TI_WITH_LLVM
std::unique_ptr<JITSession> create_llvm_jit_session_cpu(Program *prog,
                                                        Arch arch);
std::unique_ptr<JITSession> create_llvm_jit_session_cuda(Program *prog,
                                                         Arch arch);
#endif

JITSession::JITSession(Program *program) : prog(program) {
}

std::unique_ptr<JITSession> JITSession::create(Program *prog, Arch arch) {
#ifdef TI_WITH_LLVM
  if (arch_is_cpu(arch)) {
    return create_llvm_jit_session_cpu(prog, arch);
  } else if (arch == Arch::cuda) {
#if defined(TI_WITH_CUDA)
    return create_llvm_jit_session_cuda(prog, arch);
#else
    TI_NOT_IMPLEMENTED
#endif
  }
#else
  TI_ERROR("Llvm disabled");
#endif
}

#ifdef TI_WITH_LLVM
llvm::DataLayout JITSession::get_data_layout() {
  TI_NOT_IMPLEMENTED
}
#endif

TLANG_NAMESPACE_END
