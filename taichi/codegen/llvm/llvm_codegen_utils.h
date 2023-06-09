#pragma once

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"

#if defined(TI_WITH_AMDGPU)
#include "llvm/IR/IntrinsicsAMDGPU.h"
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

#include "taichi/runtime/llvm/llvm_context.h"

namespace taichi::lang {

inline constexpr char kLLVMPhysicalCoordinatesName[] = "PhysicalCoordinates";

std::string type_name(llvm::Type *type);

bool is_same_type(llvm::Type *a, llvm::Type *b);

void check_func_call_signature(llvm::FunctionType *func_type,
                               llvm::StringRef func_name,
                               std::vector<llvm::Value *> &arglist,
                               llvm::IRBuilder<> *builder);

class LLVMModuleBuilder {
 public:
  std::unique_ptr<llvm::Module> module{nullptr};
  llvm::BasicBlock *entry_block{nullptr};
  std::unique_ptr<llvm::IRBuilder<>> builder{nullptr};
  TaichiLLVMContext *tlctx{nullptr};
  llvm::LLVMContext *llvm_context{nullptr};

  LLVMModuleBuilder(std::unique_ptr<llvm::Module> &&module,
                    TaichiLLVMContext *tlctx)
      : module(std::move(module)), tlctx(tlctx) {
    TI_ASSERT(this->module != nullptr);
    TI_ASSERT(&this->module->getContext() == tlctx->get_this_thread_context());
  }

  llvm::Value *create_entry_block_alloca(llvm::Type *type,
                                         std::size_t alignment = 0,
                                         llvm::Value *array_size = nullptr) {
    llvm::IRBuilderBase::InsertPointGuard guard(*builder);
    builder->SetInsertPoint(entry_block);
    auto alloca = builder->CreateAlloca(type, (unsigned)0, array_size);
    if (alignment != 0) {
      alloca->setAlignment(llvm::Align(alignment));
    }
    return alloca;
  }

  llvm::Value *create_entry_block_alloca(DataType dt) {
    auto type = tlctx->get_data_type(dt);
    return create_entry_block_alloca(type);
  }

  llvm::Type *get_runtime_type(const std::string &name) {
    return tlctx->get_runtime_type(name);
  }

  llvm::Function *get_runtime_function(const std::string &name) {
    auto f = tlctx->get_runtime_function(name);
    if (!f) {
      TI_ERROR("LLVMRuntime function {} not found.", name);
    }
    f = llvm::cast<llvm::Function>(
        module
            ->getOrInsertFunction(name, f->getFunctionType(),
                                  f->getAttributes())
            .getCallee());
    return f;
  }

  llvm::Value *call(llvm::IRBuilder<> *builder,
                    llvm::Value *func,
                    llvm::FunctionType *func_ty,
                    std::vector<llvm::Value *> args) {
    check_func_call_signature(func_ty, func->getName(), args, builder);
    return builder->CreateCall(func_ty, func, std::move(args));
  }

  llvm::Value *call(llvm::Value *func,
                    llvm::FunctionType *func_ty,
                    std::vector<llvm::Value *> args) {
    return call(builder.get(), func, func_ty, std::move(args));
  }

  llvm::Value *call(llvm::IRBuilder<> *builder,
                    llvm::Function *func,
                    std::vector<llvm::Value *> args) {
    return call(builder, func, func->getFunctionType(), std::move(args));
  }

  llvm::Value *call(llvm::Function *func, std::vector<llvm::Value *> args) {
    return call(builder.get(), func, std::move(args));
  }

  llvm::Value *call(llvm::IRBuilder<> *builder,
                    const std::string &func_name,
                    std::vector<llvm::Value *> args) {
    auto func = get_runtime_function(func_name);
    return call(builder, func, std::move(args));
  }

  llvm::Value *call(const std::string &func_name,
                    std::vector<llvm::Value *> args) {
    return call(builder.get(), func_name, std::move(args));
  }

  template <typename... Args>
  llvm::Value *call(llvm::IRBuilder<> *builder,
                    llvm::Function *func,
                    Args *...args) {
    return call(builder, func, {args...});
  }

  template <typename... Args>
  llvm::Value *call(llvm::Function *func, Args &&...args) {
    return call(builder.get(), func, std::forward<Args>(args)...);
  }

  template <typename... Args>
  llvm::Value *call(llvm::IRBuilder<> *builder,
                    const std::string &func_name,
                    Args *...args) {
    return call(builder, func_name, {args...});
  }

  template <typename... Args>
  llvm::Value *call(const std::string &func_name, Args &&...args) {
    return call(builder.get(), func_name, std::forward<Args>(args)...);
  }
};

class RuntimeObject {
 public:
  std::string cls_name;
  llvm::Value *ptr{nullptr};
  LLVMModuleBuilder *mb{nullptr};
  llvm::Type *type{nullptr};
  llvm::IRBuilder<> *builder{nullptr};

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

  void set(const std::string &field, llvm::Value *index, llvm::Value *val) {
    call(fmt::format("set_{}", field), index, val);
  }

  template <typename... Args>
  llvm::Value *call(const std::string &func_name, Args &&...args) {
    auto func = get_func(func_name);
    auto arglist = std::vector<llvm::Value *>({ptr, args...});
    check_func_call_signature(func->getFunctionType(), func->getName(), arglist,
                              builder);
    return builder->CreateCall(func, std::move(arglist));
  }

  llvm::Function *get_func(const std::string &func_name) const {
    return mb->get_runtime_function(fmt::format("{}_{}", cls_name, func_name));
  }
};

}  // namespace taichi::lang
