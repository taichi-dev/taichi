#pragma once

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#if LLVM_VERSION_MAJOR >= 10
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#endif
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "llvm_context.h"

TLANG_NAMESPACE_BEGIN

std::string type_name(llvm::Type *type);

void check_func_call_signature(llvm::Value *func,
                               std::vector<llvm::Value *> arglist);

template <typename... Args>
inline bool check_func_call_signature(llvm::Value *func, Args &&... args) {
  return check_func_call_signature(func, {args...});
}

class LLVMModuleBuilder {
 public:
  std::unique_ptr<llvm::Module> module;
  llvm::BasicBlock *entry_block;
  std::unique_ptr<llvm::IRBuilder<>> builder;
  TaichiLLVMContext *tlctx;
  llvm::LLVMContext *llvm_context;

  LLVMModuleBuilder(std::unique_ptr<llvm::Module> &&module,
                    TaichiLLVMContext *tlctx)
      : module(std::move(module)), tlctx(tlctx) {
    TI_ASSERT(this->module != nullptr);
    TI_ASSERT(&this->module->getContext() == tlctx->get_this_thread_context());
  }

  llvm::Value *create_entry_block_alloca(llvm::Type *type,
                                         std::size_t alignment = 0) {
    llvm::IRBuilderBase::InsertPointGuard guard(*builder);
    builder->SetInsertPoint(entry_block);
    auto alloca = builder->CreateAlloca(type, (unsigned)0);
    if (alignment != 0) {
#if LLVM_VERSION_MAJOR >= 10
      alloca->setAlignment(llvm::MaybeAlign(alignment));
#else
      alloca->setAlignment(alignment);
#endif
    }
    return alloca;
  }

  llvm::Value *create_entry_block_alloca(DataType dt) {
    return create_entry_block_alloca(tlctx->get_data_type(dt));
  }

  llvm::Type *get_runtime_type(const std::string &name) {
    auto ty = module->getTypeByName("struct." + name);
    if (!ty) {
      TI_ERROR("LLVMRuntime type {} not found.", name);
    }
    return ty;
  }

  llvm::Function *get_runtime_function(const std::string &name) {
    auto f = module->getFunction(name);
    if (!f) {
      TI_ERROR("LLVMRuntime function {} not found.", name);
    }
    f->removeAttribute(llvm::AttributeList::FunctionIndex,
                       llvm::Attribute::OptimizeNone);
    f->removeAttribute(llvm::AttributeList::FunctionIndex,
                       llvm::Attribute::NoInline);
    f->addAttribute(llvm::AttributeList::FunctionIndex,
                    llvm::Attribute::AlwaysInline);
    return f;
  }

  llvm::Value *call(llvm::IRBuilder<> *builder,
                    const std::string &func_name,
                    const std::vector<llvm::Value *> &arglist) {
    auto func = get_runtime_function(func_name);
    check_func_call_signature(func, arglist);
    return builder->CreateCall(func, arglist);
  }

  template <typename... Args>
  llvm::Value *call(llvm::IRBuilder<> *builder,
                    const std::string &func_name,
                    Args &&... args) {
    auto func = get_runtime_function(func_name);
    auto arglist = std::vector<llvm::Value *>({args...});
    check_func_call_signature(func, arglist);
    return builder->CreateCall(func, arglist);
  }

  template <typename... Args>
  llvm::Value *call(const std::string &func_name, Args &&... args) {
    return call(this->builder.get(), func_name, std::forward<Args>(args)...);
  }
};

class RuntimeObject {
 public:
  std::string cls_name;
  llvm::Value *ptr;
  LLVMModuleBuilder *mb;
  llvm::Type *type;
  llvm::IRBuilder<> *builder;

  RuntimeObject(const std::string &cls_name,
                LLVMModuleBuilder *mb,
                llvm::IRBuilder<> *builder,
                llvm::Value *init = nullptr)
      : cls_name(cls_name), mb(mb), builder(builder) {
    type = mb->get_runtime_type(cls_name);
    if (init == nullptr) {
      ptr = mb->create_entry_block_alloca(type);
    } else {
      ptr = builder->CreateBitCast(init, llvm::PointerType::get(type, 0));
    }
  }

  llvm::Value *get(const std::string &field) {
    return call(fmt::format("get_{}", field));
  }

  llvm::Value *get(const std::string &field, llvm::Value *index) {
    return call(fmt::format("get_{}", field), index);
  }

  llvm::Value *get_ptr(const std::string &field) {
    return call(fmt::format("get_ptr_{}", field));
  }

  void set(const std::string &field, llvm::Value *val) {
    call(fmt::format("set_{}", field), val);
  }

  template <typename... Args>
  llvm::Value *call(const std::string &func_name, Args &&... args) {
    auto func = get_func(func_name);
    auto arglist = std::vector<llvm::Value *>({ptr, args...});
    check_func_call_signature(func, arglist);
    return builder->CreateCall(func, arglist);
  }

  llvm::Value *get_func(const std::string &func_name) const {
    return mb->get_runtime_function(fmt::format("{}_{}", cls_name, func_name));
  }
};

TLANG_NAMESPACE_END
