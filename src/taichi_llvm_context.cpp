#include <llvm/Transforms/Utils/Cloning.h>
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
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
#include "llvm/Bitcode/BitcodeReader.h"
#include <llvm/Linker/Linker.h>
#include <llvm/Demangle/Demangle.h>

#include "util.h"
#include "taichi_llvm_context.h"
#include "backends/llvm_jit.h"

TLANG_NAMESPACE_BEGIN

static llvm::ExitOnError exit_on_err;

TaichiLLVMContext::TaichiLLVMContext() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
  ctx = std::make_unique<llvm::LLVMContext>();
  jit = exit_on_err(TaichiLLVMJIT::Create());
}

TaichiLLVMContext::~TaichiLLVMContext() {
}

llvm::Type *TaichiLLVMContext::get_data_type(DataType dt) {
  if (dt == DataType::i32) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt == DataType::f32) {
    return llvm::Type::getFloatTy(*ctx);
  } else if (dt == DataType::f64) {
    return llvm::Type::getDoubleTy(*ctx);
  } else {
    TC_INFO(data_type_name(dt));
    TC_NOT_IMPLEMENTED
  }
  return nullptr;
}

std::string find_existing_command(const std::vector<std::string> &commands) {
  for (auto &cmd : commands) {
    if (command_exist(cmd)) {
      return cmd;
    }
  }
  TC_P(commands);
  TC_ERROR("No command found.");
  return "";
}

void compile_runtime() {
  auto clang = find_existing_command({"clang-7", "clang"});
  TC_ASSERT(command_exist("llvm-as"));
  TC_TRACE("Compiling runtime module bitcode...");
  auto runtime_folder = get_project_fn() + "/runtime/";
  int ret = std::system(
      fmt::format("{} -S {}runtime.cpp -o {}runtime.ll -emit-llvm -std=c++17",
                  clang, runtime_folder, runtime_folder)
          .c_str());
  if (ret) {
    TC_ERROR("Runtime compilation failed.");
  }
  std::system(fmt::format("llvm-as {}runtime.ll -o {}runtime.bc",
                          runtime_folder, runtime_folder)
                  .c_str());
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::get_init_module() {
  return clone_runtime_module();
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_runtime_module() {
  if (!runtime_module) {
    compile_runtime();
    std::ifstream ifs(get_project_fn() + "/runtime/runtime.bc");
    std::string bitcode(std::istreambuf_iterator<char>(ifs),
                        (std::istreambuf_iterator<char>()));
    auto runtime =
        parseBitcodeFile(MemoryBufferRef(bitcode, "runtime_bitcode"), *ctx);
    if (!runtime) {
      TC_ERROR("Runtime bitcode load failure.");
    }
    runtime_module = std::move(runtime.get());
    for (auto &f : *runtime_module) {
      TC_INFO("Loaded runtime function: {}", std::string(f.getName()));
    }
  }
  return llvm::CloneModule(*runtime_module);
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_struct_module() {
  TC_ASSERT(struct_module);
  /*
  auto runtime = clone_runtime_module();
  bool failed =
      llvm::Linker::linkModules(*runtime, llvm::CloneModule(*struct_module));
  if (failed) {
    TC_ERROR("Runtime linking failure.");
  }
  */
  // return llvm::CloneModule(*struct_module);
  // runtime_module->print(llvm::errs(), nullptr);
  return llvm::CloneModule(*struct_module);
}

void TaichiLLVMContext::set_struct_module(
    const std::unique_ptr<llvm::Module> &module) {
  TC_ASSERT(module);
  struct_module = llvm::CloneModule(*module);
}

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(T t) {
  if constexpr (std::is_same_v<T, float32> || std::is_same_v<T, float64>) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat(t));
  } else if (std::is_same_v<T, int32> || std::is_same_v<T, uint32>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(32, t, true));
  } else if (std::is_same_v<T, int64> || std::is_same_v<T, uint64>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(64, t, true));
  } else {
    TC_NOT_IMPLEMENTED
    return nullptr;
  }
}

std::string TaichiLLVMContext::type_name(llvm::Type *type) {
  std::string type_name;
  llvm::raw_string_ostream rso(type_name);
  type->print(rso);
  return rso.str();
}

std::size_t TaichiLLVMContext::get_type_size(llvm::Type *type) {
  return jit->get_type_size(type);
}

template llvm::Value *TaichiLLVMContext::get_constant(float32 t);
template llvm::Value *TaichiLLVMContext::get_constant(float64 t);

template llvm::Value *TaichiLLVMContext::get_constant(int32 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint32 t);

template llvm::Value *TaichiLLVMContext::get_constant(int64 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint64 t);

TLANG_NAMESPACE_END
