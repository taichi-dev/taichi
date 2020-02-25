// A LLVM JIT compiler wrapper
#pragma once

// Based on
// https://llvm.org/docs/tutorial/BuildingAJIT3.html

// Note that
// https://llvm.org/docs/tutorial/BuildingAJIT2.html
// leads to a JIT that crashes all C++ exception after JIT session
// destruction...

#if defined(min)
#undef min
#endif
#if defined(max)
#undef max
#endif
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/CompileOnDemandLayer.h"
#include "llvm/ExecutionEngine/Orc/CompileUtils.h"
#include "llvm/ExecutionEngine/Orc/IRCompileLayer.h"
#include "llvm/ExecutionEngine/Orc/IRTransformLayer.h"
#include "llvm/ExecutionEngine/Orc/LambdaResolver.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/RTDyldMemoryManager.h"
#include "llvm/ExecutionEngine/RuntimeDyld.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/IPO.h"
#include <memory>
#include "../tlang_util.h"
#include "jit_session.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
using namespace llvm::orc;

std::string compile_module_to_ptx(std::unique_ptr<llvm::Module> &module);
int compile_ptx_and_launch(const std::string &ptx,
                           const std::string &kernel_name,
                           void *);

TLANG_NAMESPACE_END
