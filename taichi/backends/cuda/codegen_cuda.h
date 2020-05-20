/*******************************************************************************
     Copyright (c) 2020 The Taichi Authors
     Use of this software is governed by the LICENSE file.
*******************************************************************************/

// The CUDA backend

#pragma once

#include "taichi/codegen/codegen.h"

TLANG_NAMESPACE_BEGIN

class CodeGenCUDA : public KernelCodeGen {
 public:
  CodeGenCUDA(Kernel *kernel, IRNode *ir = nullptr)
      : KernelCodeGen(kernel, ir) {
  }

  virtual FunctionType codegen() override;
};

TLANG_NAMESPACE_END
