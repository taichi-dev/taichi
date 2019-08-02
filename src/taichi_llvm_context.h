#pragma once
#include "util.h"

namespace llvm {
  class LLVMContext;
  class Type;
  class Value;
}

TLANG_NAMESPACE_BEGIN

class TaichiLLVMJIT;

class TaichiLLVMContext {
 public:
  std::unique_ptr<llvm::LLVMContext> ctx;
  std::unique_ptr<TaichiLLVMJIT> jit;

  llvm::Type *get_data_type(DataType dt);

  TaichiLLVMContext();

  ~TaichiLLVMContext();
};

TLANG_NAMESPACE_END
