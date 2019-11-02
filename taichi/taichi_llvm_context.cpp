// A helper for the llvm backend

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
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include <llvm/Linker/Linker.h>
#include <llvm/Demangle/Demangle.h>

#include "tlang_util.h"
#include "taichi_llvm_context.h"
#include "backends/llvm_jit.h"

TLANG_NAMESPACE_BEGIN

static llvm::ExitOnError exit_on_err;

TaichiLLVMContext::TaichiLLVMContext(Arch arch) : arch(arch) {
  llvm::InitializeAllTargets();
  llvm::remove_fatal_error_handler();
  llvm::install_fatal_error_handler(
      [](void *user_data, const std::string &reason, bool gen_crash_diag) {
        TC_ERROR("LLVM Fatal Error: {}", reason);
      },
      nullptr);

  if (arch == Arch::x86_64) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  } else {
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXAsmPrinter();
  }
  ctx = std::make_unique<llvm::LLVMContext>();
  TC_INFO("Creating llvm context for arch: {}", arch_name(arch));
  jit = exit_on_err(TaichiLLVMJIT::create(arch));
}

TaichiLLVMContext::~TaichiLLVMContext() {
}

llvm::Type *TaichiLLVMContext::get_data_type(DataType dt) {
  if (dt == DataType::i32) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt == DataType::i8) {
    return llvm::Type::getInt8Ty(*ctx);
  } else if (dt == DataType::i16) {
    return llvm::Type::getInt16Ty(*ctx);
  } else if (dt == DataType::i64) {
    return llvm::Type::getInt64Ty(*ctx);
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
}

std::string get_runtime_fn(Arch arch) {
  return fmt::format("runtime_{}.bc", arch_name(arch));
}

std::string get_runtime_dir() {
  return get_repo_dir() + "/taichi/runtime/";
}

void compile_runtime_bitcode(Arch arch) {
  if (is_release())
    return;
  TI_AUTO_PROF;
  static std::set<int> runtime_compiled;
  if (runtime_compiled.find((int)arch) == runtime_compiled.end()) {
    auto clang = find_existing_command({"clang-7", "clang"});
    TC_ASSERT(command_exist("llvm-as"));
    TC_TRACE("Compiling runtime module bitcode...");
    auto runtime_folder = get_runtime_dir();
    std::string macro = fmt::format(" -D ARCH_{} ", arch_name(arch));
    int ret = std::system(
        fmt::format(
            "{} -S {}runtime.cpp -o {}runtime.ll -emit-llvm -std=c++17 {}",
            clang, runtime_folder, runtime_folder, macro)
            .c_str());
    if (ret) {
      TC_ERROR("Runtime compilation failed.");
    }
    std::system(fmt::format("llvm-as {}runtime.ll -o {}{}", runtime_folder,
                            runtime_folder, get_runtime_fn(arch))
                    .c_str());
    runtime_compiled.insert((int)arch);
  }
}

void compile_runtimes() {
  compile_runtime_bitcode(Arch::x86_64);
#if defined(TLANG_WITH_CUDA)
  compile_runtime_bitcode(Arch::gpu);
#endif
}

std::string libdevice_path() {
#if defined(TLANG_WITH_CUDA)
  auto folder =
      fmt::format("/usr/local/cuda-{}/nvvm/libdevice/", TLANG_CUDA_VERSION);
  if (is_release()) {
    folder = compiled_lib_dir;
  }
  auto cuda_version_major = int(std::atof(TLANG_CUDA_VERSION));
  return fmt::format("{}/libdevice.{}.bc", folder, cuda_version_major);
#else
  TC_NOT_IMPLEMENTED;
  return "";
#endif
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::get_init_module() {
  return clone_runtime_module();
}

std::unique_ptr<llvm::Module> module_from_bitcode_file(std::string bitcode_path,
                                                       llvm::LLVMContext *ctx) {
  TI_AUTO_PROF
  std::ifstream ifs(bitcode_path);
  std::string bitcode(std::istreambuf_iterator<char>(ifs),
                      (std::istreambuf_iterator<char>()));
  auto runtime =
      parseBitcodeFile(MemoryBufferRef(bitcode, "runtime_bitcode"), *ctx);
  if (!runtime) {
    TC_ERROR("Runtime bitcode load failure.");
  }
  bool module_broken = llvm::verifyModule(*runtime.get(), &llvm::errs());
  TC_ERROR_IF(module_broken, "Module broken");
  return std::move(runtime.get());
}

int num_instructions(llvm::Function *func) {
  int counter = 0;
  for (BasicBlock &bb : *func)
    counter += std::distance(bb.begin(), bb.end());
  return counter;
}

void remove_useless_libdevice_functions(llvm::Module *module) {
  std::vector<std::string> function_name_list = {
      "rnorm3df",
      "norm4df",
      "rnorm4df",
      "normf",
      "rnormf",
      "j0f",
      "j1f",
      "y0f",
      "y1f",
      "ynf",
      "jnf",
      "cyl_bessel_i0f",
      "cyl_bessel_i1f",
      "j0",
      "j1",
      "y0",
      "y1",
      "yn",
      "jn",
      "cyl_bessel_i0",
      "cyl_bessel_i1",
      "tgammaf",
      "lgammaf",
      "tgamma",
      "lgamma",
      "erff",
      "erfinvf",
      "erfcf",
      "erfcxf",
      "erfcinvf",
      "erf",
      "erfinv",
      "erfcx",
      "erfcinv",
      "erfc",
  };
  for (auto fn : function_name_list) {
    module->getFunction("__nv_" + fn)->eraseFromParent();
  }
  module->getFunction("__internal_lgamma_pos")->eraseFromParent();
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_runtime_module() {
  TI_AUTO_PROF
  if (!runtime_module) {
    if (is_release()) {
      runtime_module = module_from_bitcode_file(
          fmt::format("{}/{}", compiled_lib_dir, get_runtime_fn(arch)),
          ctx.get());
    } else {
      compile_runtime_bitcode(arch);
      runtime_module = module_from_bitcode_file(
          fmt::format("{}/{}", get_runtime_dir(), get_runtime_fn(arch)),
          ctx.get());
    }
    if (arch == Arch::gpu) {
      runtime_module->setTargetTriple("nvptx64-nvidia-cuda");

      auto patch_intrinsic = [&](std::string name, Intrinsic::ID intrin,
                                 bool ret = true) {
        TC_PROFILER("patch intrinsic");
        auto func = runtime_module->getFunction(name);
        func->getEntryBlock().eraseFromParent();
        auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
        IRBuilder<> builder(*ctx);
        builder.SetInsertPoint(bb);
        if (ret) {
          builder.CreateRet(builder.CreateIntrinsic(intrin, {}, {}));
        } else {
          builder.CreateIntrinsic(intrin, {}, {});
          builder.CreateRetVoid();
        }
      };

      patch_intrinsic("thread_idx", Intrinsic::nvvm_read_ptx_sreg_tid_x);
      patch_intrinsic("block_idx", Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
      patch_intrinsic("block_dim", Intrinsic::nvvm_read_ptx_sreg_ntid_x);
      patch_intrinsic("grid_dim", Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
      patch_intrinsic("block_barrier", Intrinsic::nvvm_barrier0, false);

      link_module_with_libdevice(runtime_module);

      // runtime_module->print(llvm::errs(), nullptr);
    }

    /*
    int total_inst = 0;
    int total_big_inst = 0;

    for (auto &f : *runtime_module) {
      int c = num_instructions(&f);
      if (c > 100) {
        total_big_inst += c;
        TC_INFO("Loaded runtime function: {} (inst. count= {})",
                std::string(f.getName()), c);
      }
      total_inst += c;
    }
    TC_P(total_inst);
    TC_P(total_big_inst);
    */
  }
  std::unique_ptr<llvm::Module> cloned;
  {
    TC_PROFILER("clone module");
    cloned = llvm::CloneModule(*runtime_module);
  }
  return cloned;
}

void TaichiLLVMContext::link_module_with_libdevice(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  auto libdevice_module = module_from_bitcode_file(libdevice_path(), ctx.get());

  remove_useless_libdevice_functions(libdevice_module.get());

  std::vector<std::string> libdevice_function_names;
  for (auto &f : *libdevice_module) {
    if (!f.isDeclaration()) {
      libdevice_function_names.push_back(f.getName());
    }
  }

  libdevice_module->setTargetTriple("nvptx64-nvidia-cuda");
  module->setDataLayout(libdevice_module->getDataLayout());

  bool failed = llvm::Linker::linkModules(*module, std::move(libdevice_module));
  if (failed) {
    TC_ERROR("CUDA libdevice linking failure.");
  }

  for (auto func_name : libdevice_function_names) {
    auto func = module->getFunction(func_name);
    if (!func) {
      TC_P(func_name);
    } else
      func->setLinkage(Function::InternalLinkage);
  }
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_struct_module() {
  TC_ASSERT(struct_module);
  return llvm::CloneModule(*struct_module);
}

void TaichiLLVMContext::set_struct_module(
    const std::unique_ptr<llvm::Module> &module) {
  TC_ASSERT(module);
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TC_ERROR("module broken");
  }
  struct_module = llvm::CloneModule(*module);
}

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(DataType dt, T t) {
  if (dt == DataType::f32) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat((float32)t));
  } else if (dt == DataType::f64) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat((float64)t));
  } else if (dt == DataType::i32) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(32, t, true));
  } else if (dt == DataType::u32) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(32, t, false));
  } else if (dt == DataType::i64) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(64, t, true));
  } else if (dt == DataType::u64) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(64, t, false));
  } else {
    TC_NOT_IMPLEMENTED
    return nullptr;
  }
}

template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, int32 t);

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(T t) {
  using TargetType = T;
  if constexpr (std::is_same_v<TargetType, float32> ||
                std::is_same_v<TargetType, float64>) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat(t));
  } else if (std::is_same_v<TargetType, bool>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(1, t, false));
  } else if (std::is_same_v<TargetType, int32> ||
             std::is_same_v<TargetType, uint32>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(32, t, true));
  } else if (std::is_same_v<TargetType, int64> ||
             std::is_same_v<TargetType, uint64>) {
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

template llvm::Value *TaichiLLVMContext::get_constant(bool t);

template llvm::Value *TaichiLLVMContext::get_constant(int32 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint32 t);

template llvm::Value *TaichiLLVMContext::get_constant(int64 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint64 t);

TLANG_NAMESPACE_END
