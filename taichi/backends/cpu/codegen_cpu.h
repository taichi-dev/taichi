/*******************************************************************************
     Copyright (c) 2020 The Taichi Authors
     Use of this software is governed by the LICENSE file.
*******************************************************************************/

// x86 backend implementation

#pragma once

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCPU : public KernelCodeGen {
 public:
  CodeGenCPU(Kernel *kernel, IRNode *ir = nullptr) : KernelCodeGen(kernel, ir) {
  }

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
