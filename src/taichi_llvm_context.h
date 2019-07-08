#pragma once
#include "util.h"

namespace llvm {
  class LLVMContext;
}

TLANG_NAMESPACE_BEGIN

class TaichiLLVMJIT;

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<TaichiLLVMJIT> jit;

  TaichiLLVMContext();

  ~TaichiLLVMContext();
};

TLANG_NAMESPACE_END
