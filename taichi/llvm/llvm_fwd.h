/*******************************************************************************
     Copyright (c) 2020 The Taichi Authors
     Use of this software is governed by the LICENSE file.
*******************************************************************************/

#pragma once

namespace llvm {
class LLVMContext;
class Type;
class Value;
class Module;
class Function;
class DataLayout;
class JITSymbol;
class ExitOnError;
namespace orc {
class ThreadSafeContext;
}
}  // namespace llvm
