#include "jit_session.h"

TLANG_NAMESPACE_BEGIN

class JITSessionCUDA : public JITSession {
 public:
  JITSessionCUDA() {
  }

  /*
  void add_module(std::unique_ptr<llvm::Module> &&module) override {
  }
   */
};

TLANG_NAMESPACE_END
