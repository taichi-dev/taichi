// A llvm backend helper

#include "taichi/llvm/llvm_context.h"

#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/LLVMContext.h"
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
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "taichi/lang_util.h"
#include "taichi/jit/jit_session.h"
#include "taichi/common/task.h"
#include "taichi/util/environ_config.h"
#include "llvm_context.h"

#ifdef _WIN32
// Travis CI seems doesn't support <filesystem>...
#include <filesystem>
#else
#include <unistd.h>
#endif

#if defined(TI_WITH_CUDA)
#include "taichi/backends/cuda/cuda_context.h"
#endif

namespace taichi {
namespace lang {

using namespace llvm;

TaichiLLVMContext::TaichiLLVMContext(LlvmProgramImpl *llvm_prog, Arch arch)
    : arch_(arch) {
  TI_TRACE("Creating Taichi llvm context for arch: {}", arch_name(arch));
  main_thread_id_ = std::this_thread::get_id();
  main_thread_data_ = get_this_thread_data();
  llvm::remove_fatal_error_handler();
  llvm::install_fatal_error_handler(
      [](void *user_data, const std::string &reason, bool gen_crash_diag) {
        TI_ERROR("LLVM Fatal Error: {}", reason);
      },
      nullptr);

  if (arch_is_cpu(arch)) {
#if defined(TI_PLATFORM_OSX) and defined(TI_ARCH_ARM)
    // Note that on Apple Silicon (M1), "native" seems to mean arm instead of
    // arm64 (aka AArch64).
    LLVMInitializeAArch64Target();
    LLVMInitializeAArch64TargetMC();
    LLVMInitializeAArch64TargetInfo();
    LLVMInitializeAArch64AsmPrinter();
#else
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
#endif
  } else {
#if defined(TI_WITH_CUDA)
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXAsmPrinter();
#else
    TI_NOT_IMPLEMENTED
#endif
  }
  jit = JITSession::create(llvm_prog, arch);
  TI_TRACE("Taichi llvm context created.");
}

TaichiLLVMContext::~TaichiLLVMContext() {
}

llvm::Type *TaichiLLVMContext::get_data_type(DataType dt) {
  auto ctx = get_this_thread_context();
  if (dt->is_primitive(PrimitiveTypeID::i32)) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::i8)) {
    return llvm::Type::getInt8Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::i16)) {
    return llvm::Type::getInt16Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::i64)) {
    return llvm::Type::getInt64Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return llvm::Type::getFloatTy(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return llvm::Type::getDoubleTy(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::u8)) {
    return llvm::Type::getInt8Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::u16)) {
    return llvm::Type::getInt16Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::u32)) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::u64)) {
    return llvm::Type::getInt64Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    return llvm::Type::getHalfTy(*ctx);
  } else {
    TI_INFO(data_type_name(dt));
    TI_NOT_IMPLEMENTED
  }
}

std::string find_existing_command(const std::vector<std::string> &commands) {
  for (auto &cmd : commands) {
    if (command_exist(cmd)) {
      return cmd;
    }
  }
  for (const auto &cmd : commands) {
    TI_WARN("Potential command {}", cmd);
  }
  TI_ERROR("None command found.");
}

std::string get_runtime_fn(Arch arch) {
  return fmt::format("runtime_{}.bc", arch_name(arch));
}

std::string libdevice_path() {
  std::string folder;
  folder = runtime_lib_dir();
  auto cuda_version_string = get_cuda_version_string();
  auto cuda_version_major = int(std::atof(cuda_version_string.c_str()));
  return fmt::format("{}/slim_libdevice.{}.bc", folder, cuda_version_major);
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_module_to_context(
    llvm::Module *module,
    llvm::LLVMContext *target_context) {
  // Dump a module from one context to bitcode and then parse the bitcode in a
  // different context
  std::string bitcode;

  {
    std::lock_guard<std::mutex> _(mut_);
    llvm::raw_string_ostream sos(bitcode);
    // Use a scope to make sure sos flushes on destruction
    llvm::WriteBitcodeToFile(*module, sos);
  }

  auto cloned = parseBitcodeFile(
      llvm::MemoryBufferRef(bitcode, "runtime_bitcode"), *target_context);
  if (!cloned) {
    auto error = cloned.takeError();
    TI_ERROR("Bitcode cloned failed.");
  }
  return std::move(cloned.get());
}

std::unique_ptr<llvm::Module>
TaichiLLVMContext::clone_module_to_this_thread_context(llvm::Module *module) {
  TI_TRACE("Cloning struct module");
  TI_ASSERT(module);
  auto this_context = get_this_thread_context();
  return clone_module_to_context(module, this_context);
}

std::unique_ptr<llvm::Module> LlvmModuleBitcodeLoader::load(
    llvm::LLVMContext *ctx) const {
  TI_AUTO_PROF;
  std::ifstream ifs(bitcode_path_, std::ios::binary);
  TI_ERROR_IF(!ifs, "Bitcode file ({}) not found.", bitcode_path_);
  std::string bitcode(std::istreambuf_iterator<char>(ifs),
                      (std::istreambuf_iterator<char>()));
  auto runtime =
      parseBitcodeFile(llvm::MemoryBufferRef(bitcode, buffer_id_), *ctx);
  if (!runtime) {
    auto error = runtime.takeError();
    TI_WARN("Bitcode loading error message:");
    llvm::errs() << error << "\n";
    TI_ERROR("Failed to load bitcode={}", bitcode_path_);
    return nullptr;
  }

  if (inline_funcs_) {
    for (auto &f : *(runtime.get())) {
      TaichiLLVMContext::mark_inline(&f);
    }
  }

  const bool module_broken = llvm::verifyModule(*runtime.get(), &llvm::errs());
  if (module_broken) {
    TI_ERROR("Broken bitcode={}", bitcode_path_);
    return nullptr;
  }
  return std::move(runtime.get());
}

std::unique_ptr<llvm::Module> module_from_bitcode_file(
    const std::string &bitcode_path,
    llvm::LLVMContext *ctx) {
  LlvmModuleBitcodeLoader loader;
  return loader.set_bitcode_path(bitcode_path)
      .set_buffer_id("runtime_bitcode")
      .set_inline_funcs(true)
      .load(ctx);
}

// The goal of this function is to rip off huge libdevice functions that are not
// going to be used later, at an early stage. Although the LLVM optimizer will
// ultimately remove unused functions during a global DCE pass, we don't even
// want these functions to waste clock cycles during module cloning and linking.
static void remove_useless_cuda_libdevice_functions(llvm::Module *module) {
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

void TaichiLLVMContext::init_runtime_jit_module() {
  update_runtime_jit_module(clone_runtime_module());
}

// Note: runtime_module = init_module < struct_module

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_runtime_module() {
  TI_AUTO_PROF
  TI_ASSERT(std::this_thread::get_id() == main_thread_id_);
  auto data = get_this_thread_data();
  if (!data->runtime_module) {
    data->runtime_module = clone_module(get_runtime_fn(arch_));
  }

  std::unique_ptr<llvm::Module> cloned;
  {
    TI_PROFILER("clone module");
    cloned = llvm::CloneModule(*data->runtime_module);
  }

  TI_ASSERT(cloned != nullptr);

  return cloned;
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_module(
    const std::string &file) {
  auto ctx = get_this_thread_context();
  std::unique_ptr<llvm::Module> module = module_from_bitcode_file(
      fmt::format("{}/{}", runtime_lib_dir(), file), ctx);
  if (arch_ == Arch::cuda) {
    module->setTargetTriple("nvptx64-nvidia-cuda");

#if defined(TI_WITH_CUDA)
    auto func = module->getFunction("cuda_compute_capability");
    if (func) {
      func->deleteBody();
      auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
      IRBuilder<> builder(*ctx);
      builder.SetInsertPoint(bb);
      builder.CreateRet(
          get_constant(CUDAContext::get_instance().get_compute_capability()));
      TaichiLLVMContext::mark_inline(func);
    }
#endif

    auto patch_intrinsic = [&](std::string name, Intrinsic::ID intrin,
                               bool ret = true,
                               std::vector<llvm::Type *> types = {},
                               std::vector<llvm::Value *> extra_args = {}) {
      auto func = module->getFunction(name);
      if (!func) {
        return;
      }
      func->deleteBody();
      auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
      IRBuilder<> builder(*ctx);
      builder.SetInsertPoint(bb);
      std::vector<llvm::Value *> args;
      for (auto &arg : func->args())
        args.push_back(&arg);
      args.insert(args.end(), extra_args.begin(), extra_args.end());
      if (ret) {
        builder.CreateRet(builder.CreateIntrinsic(intrin, types, args));
      } else {
        builder.CreateIntrinsic(intrin, types, args);
        builder.CreateRetVoid();
      }
      TaichiLLVMContext::mark_inline(func);
    };

    auto patch_atomic_add = [&](std::string name,
                                llvm::AtomicRMWInst::BinOp op) {
      auto func = module->getFunction(name);
      if (!func) {
        return;
      }
      func->deleteBody();
      auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
      IRBuilder<> builder(*ctx);
      builder.SetInsertPoint(bb);
      std::vector<llvm::Value *> args;
      for (auto &arg : func->args())
        args.push_back(&arg);
      builder.CreateRet(builder.CreateAtomicRMW(
          op, args[0], args[1], llvm::AtomicOrdering::SequentiallyConsistent));
      TaichiLLVMContext::mark_inline(func);
    };

    patch_intrinsic("thread_idx", Intrinsic::nvvm_read_ptx_sreg_tid_x);
    patch_intrinsic("cuda_clock_i64", Intrinsic::nvvm_read_ptx_sreg_clock64);
    patch_intrinsic("block_idx", Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
    patch_intrinsic("block_dim", Intrinsic::nvvm_read_ptx_sreg_ntid_x);
    patch_intrinsic("grid_dim", Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
    patch_intrinsic("block_barrier", Intrinsic::nvvm_barrier0, false);
    patch_intrinsic("warp_barrier", Intrinsic::nvvm_bar_warp_sync, false);
    patch_intrinsic("block_memfence", Intrinsic::nvvm_membar_cta, false);
    patch_intrinsic("grid_memfence", Intrinsic::nvvm_membar_gl, false);
    patch_intrinsic("system_memfence", Intrinsic::nvvm_membar_sys, false);

    patch_intrinsic("cuda_all", Intrinsic::nvvm_vote_all);
    patch_intrinsic("cuda_all_sync", Intrinsic::nvvm_vote_all_sync);

    patch_intrinsic("cuda_any", Intrinsic::nvvm_vote_any);
    patch_intrinsic("cuda_any_sync", Intrinsic::nvvm_vote_any_sync);

    patch_intrinsic("cuda_uni", Intrinsic::nvvm_vote_uni);
    patch_intrinsic("cuda_uni_sync", Intrinsic::nvvm_vote_uni_sync);

    patch_intrinsic("cuda_ballot", Intrinsic::nvvm_vote_ballot);
    patch_intrinsic("cuda_ballot_sync", Intrinsic::nvvm_vote_ballot_sync);

    patch_intrinsic("cuda_shfl_down_sync_i32",
                    Intrinsic::nvvm_shfl_sync_down_i32);
    patch_intrinsic("cuda_shfl_down_sync_f32",
                    Intrinsic::nvvm_shfl_sync_down_f32);

    patch_intrinsic("cuda_shfl_up_sync_i32", Intrinsic::nvvm_shfl_sync_up_i32);
    patch_intrinsic("cuda_shfl_up_sync_f32", Intrinsic::nvvm_shfl_sync_up_f32);

    patch_intrinsic("cuda_shfl_sync_i32", Intrinsic::nvvm_shfl_sync_idx_i32);

    patch_intrinsic("cuda_shfl_sync_f32", Intrinsic::nvvm_shfl_sync_idx_f32);

    patch_intrinsic("cuda_shfl_xor_sync_i32",
                    Intrinsic::nvvm_shfl_sync_bfly_i32);

    patch_intrinsic("cuda_match_any_sync_i32",
                    Intrinsic::nvvm_match_any_sync_i32);

    // LLVM 10.0.0 seems to have a bug on this intrinsic function
    /*
    nvvm_match_all_sync_i32
    Args:
        1. u32 mask
        2. i32 value
        3. i32 *pred
    */
    /*
    patch_intrinsic("cuda_match_all_sync_i32p",
                    Intrinsic::nvvm_math_all_sync_i32);
    */

    // LLVM 10.0.0 seems to have a bug on this intrinsic function
    /*
    patch_intrinsic("cuda_match_any_sync_i64",
                    Intrinsic::nvvm_match_any_sync_i64);
                    */

    patch_intrinsic("ctlz_i32", Intrinsic::ctlz, true,
                    {llvm::Type::getInt32Ty(*ctx)}, {get_constant(false)});
    patch_intrinsic("cttz_i32", Intrinsic::cttz, true,
                    {llvm::Type::getInt32Ty(*ctx)}, {get_constant(false)});

    patch_atomic_add("atomic_add_i32", llvm::AtomicRMWInst::Add);

    patch_atomic_add("atomic_add_i64", llvm::AtomicRMWInst::Add);

    patch_atomic_add("atomic_add_f32", llvm::AtomicRMWInst::FAdd);

    patch_atomic_add("atomic_add_f64", llvm::AtomicRMWInst::FAdd);

    patch_intrinsic("block_memfence", Intrinsic::nvvm_membar_cta, false);

    link_module_with_cuda_libdevice(module);

    // To prevent potential symbol name conflicts, we use "cuda_vprintf"
    // instead of "vprintf" in llvm/runtime.cpp. Now we change it back for
    // linking
    for (auto &f : *module) {
      if (f.getName() == "cuda_vprintf") {
        f.setName("vprintf");
      }
    }

    // runtime_module->print(llvm::errs(), nullptr);
  }

  return module;
}

void TaichiLLVMContext::link_module_with_cuda_libdevice(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  TI_ASSERT(arch_ == Arch::cuda);

  auto libdevice_module =
      module_from_bitcode_file(libdevice_path(), get_this_thread_context());

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
    TI_ERROR("CUDA libdevice linking failure.");
  }

  // Make sure all libdevice functions are linked, and set their linkage to
  // internal
  for (auto func_name : libdevice_function_names) {
    auto func = module->getFunction(func_name);
    if (!func) {
      TI_INFO("Function {} not found", func_name);
    } else
      func->setLinkage(llvm::Function::InternalLinkage);
  }
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_struct_module() {
  TI_AUTO_PROF
  auto struct_module = get_this_thread_struct_module();
  TI_ASSERT(struct_module);
  return llvm::CloneModule(*struct_module);
}

void TaichiLLVMContext::set_struct_module(
    const std::unique_ptr<llvm::Module> &module) {
  auto data = get_this_thread_data();
  TI_ASSERT(module);
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("module broken");
  }
  // TODO: Move this after ``if (!arch_is_cpu(arch))``.
  data->struct_module = llvm::CloneModule(*module);
}

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(DataType dt, T t) {
  auto ctx = get_this_thread_context();
  if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat((float32)t));
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    return llvm::ConstantFP::get(llvm::Type::getHalfTy(*ctx), (float32)t);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat((float64)t));
  } else if (is_integral(dt)) {
    if (is_signed(dt)) {
      return llvm::ConstantInt::get(
          *ctx, llvm::APInt(data_type_bits(dt), (uint64_t)t, true));
    } else {
      return llvm::ConstantInt::get(
          *ctx, llvm::APInt(data_type_bits(dt), (uint64_t)t, false));
    }
  } else {
    TI_NOT_IMPLEMENTED
  }
}

template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, int32 t);
template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, int64 t);
template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, uint32 t);
template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, uint64 t);
template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, float32 t);
template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, float64 t);

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(T t) {
  auto ctx = get_this_thread_context();
  TI_ASSERT(ctx != nullptr);
  using TargetType = T;
  if constexpr (std::is_same_v<TargetType, float32> ||
                std::is_same_v<TargetType, float64>) {
    return llvm::ConstantFP::get(*ctx, llvm::APFloat(t));
  } else if (std::is_same_v<TargetType, bool>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(1, (uint64)t, true));
  } else if (std::is_same_v<TargetType, int32> ||
             std::is_same_v<TargetType, uint32>) {
    return llvm::ConstantInt::get(*ctx, llvm::APInt(32, (uint64)t, true));
  } else if (std::is_same_v<TargetType, int64> ||
             std::is_same_v<TargetType, std::size_t> ||
             std::is_same_v<TargetType, uint64>) {
    static_assert(sizeof(std::size_t) == sizeof(uint64));
    return llvm::ConstantInt::get(*ctx, llvm::APInt(64, (uint64)t, true));
  } else {
    TI_NOT_IMPLEMENTED
  }
}

std::string TaichiLLVMContext::type_name(llvm::Type *type) {
  std::string type_name;
  llvm::raw_string_ostream rso(type_name);
  type->print(rso);
  return rso.str();
}

std::size_t TaichiLLVMContext::get_type_size(llvm::Type *type) {
  return get_data_layout().getTypeAllocSize(type);
}

std::size_t TaichiLLVMContext::get_struct_element_offset(llvm::StructType *type,
                                                         int idx) {
  return get_data_layout().getStructLayout(type)->getElementOffset(idx);
}

void TaichiLLVMContext::mark_inline(llvm::Function *f) {
  for (auto &B : *f)
    for (auto &I : B) {
      if (auto *call = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (auto func = call->getCalledFunction();
            func && func->getName() == "mark_force_no_inline") {
          // Found "mark_force_no_inline". Do not inline.
          return;
        }
      }
    }
  f->removeAttribute(llvm::AttributeList::FunctionIndex,
                     llvm::Attribute::OptimizeNone);
  f->removeAttribute(llvm::AttributeList::FunctionIndex,
                     llvm::Attribute::NoInline);
  f->addAttribute(llvm::AttributeList::FunctionIndex,
                  llvm::Attribute::AlwaysInline);
}

int TaichiLLVMContext::num_instructions(llvm::Function *func) {
  int counter = 0;
  for (BasicBlock &bb : *func)
    counter += std::distance(bb.begin(), bb.end());
  return counter;
}

void TaichiLLVMContext::print_huge_functions(llvm::Module *module) {
  int total_inst = 0;
  int total_big_inst = 0;

  for (auto &f : *module) {
    int c = num_instructions(&f);
    if (c > 100) {
      total_big_inst += c;
      TI_INFO("{}: {} inst.", std::string(f.getName()), c);
    }
    total_inst += c;
  }
  TI_P(total_inst);
  TI_P(total_big_inst);
}

llvm::DataLayout TaichiLLVMContext::get_data_layout() {
  return jit->get_data_layout();
}

JITModule *TaichiLLVMContext::add_module(std::unique_ptr<llvm::Module> module) {
  return jit->add_module(std::move(module));
}

void TaichiLLVMContext::insert_nvvm_annotation(llvm::Function *func,
                                               std::string key,
                                               int val) {
  /*******************************************************************
  Example annotation from llvm PTX doc:

  define void @kernel(float addrspace(1)* %A,
                      float addrspace(1)* %B,
                      float addrspace(1)* %C);

  !nvvm.annotations = !{!0}
  !0 = !{void (float addrspace(1)*,
               float addrspace(1)*,
               float addrspace(1)*)* @kernel, !"kernel", i32 1}
  *******************************************************************/
  auto ctx = get_this_thread_context();
  llvm::Metadata *md_args[] = {llvm::ValueAsMetadata::get(func),
                               MDString::get(*ctx, key),
                               llvm::ValueAsMetadata::get(get_constant(val))};

  MDNode *md_node = MDNode::get(*ctx, md_args);

  func->getParent()
      ->getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(md_node);
}

void TaichiLLVMContext::mark_function_as_cuda_kernel(llvm::Function *func,
                                                     int block_dim) {
  // Mark kernel function as a CUDA __global__ function
  // Add the nvvm annotation that it is considered a kernel function.
  insert_nvvm_annotation(func, "kernel", 1);
  if (block_dim != 0) {
    // CUDA launch bounds
    insert_nvvm_annotation(func, "maxntidx", block_dim);
    insert_nvvm_annotation(func, "minctasm", 2);
  }
}

void TaichiLLVMContext::eliminate_unused_functions(
    llvm::Module *module,
    std::function<bool(const std::string &)> export_indicator) {
  TI_AUTO_PROF
  using namespace llvm;
  TI_ASSERT(module);
  if (false) {
    // temporary fix for now to make LLVM 8 work with CUDA
    // TODO: recover this when it's time
    if (llvm::verifyModule(*module, &llvm::errs())) {
      TI_ERROR("Module broken\n");
    }
  }
  llvm::ModulePassManager manager;
  llvm::ModuleAnalysisManager ana;
  llvm::PassBuilder pb;
  pb.registerModuleAnalyses(ana);
  manager.addPass(llvm::InternalizePass([&](const GlobalValue &val) -> bool {
    return export_indicator(val.getName());
  }));
  manager.addPass(GlobalDCEPass());
  manager.run(*module, ana);
}

TaichiLLVMContext::ThreadLocalData *TaichiLLVMContext::get_this_thread_data() {
  std::lock_guard<std::mutex> _(thread_map_mut_);
  auto tid = std::this_thread::get_id();
  if (per_thread_data_.find(tid) == per_thread_data_.end()) {
    std::stringstream ss;
    ss << tid;
    TI_TRACE("Creating thread local data for thread {}", ss.str());
    per_thread_data_[tid] = std::make_unique<ThreadLocalData>();
  }
  return per_thread_data_[tid].get();
}

llvm::LLVMContext *TaichiLLVMContext::get_this_thread_context() {
  ThreadLocalData *data = get_this_thread_data();
  if (!data->llvm_context) {
    auto ctx = std::make_unique<llvm::LLVMContext>();
    data->llvm_context = ctx.get();
    data->thread_safe_llvm_context =
        std::make_unique<llvm::orc::ThreadSafeContext>(std::move(ctx));
  }
  return data->llvm_context;
}

llvm::orc::ThreadSafeContext *
TaichiLLVMContext::get_this_thread_thread_safe_context() {
  get_this_thread_context();  // make sure the context is created
  ThreadLocalData *data = get_this_thread_data();
  return data->thread_safe_llvm_context.get();
}

llvm::Module *TaichiLLVMContext::get_this_thread_struct_module() {
  ThreadLocalData *data = get_this_thread_data();
  if (!data->struct_module) {
    data->struct_module = clone_module_to_this_thread_context(
        main_thread_data_->struct_module.get());
  }
  return data->struct_module.get();
}

template llvm::Value *TaichiLLVMContext::get_constant(float32 t);
template llvm::Value *TaichiLLVMContext::get_constant(float64 t);

template llvm::Value *TaichiLLVMContext::get_constant(bool t);

template llvm::Value *TaichiLLVMContext::get_constant(int32 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint32 t);

template llvm::Value *TaichiLLVMContext::get_constant(int64 t);
template llvm::Value *TaichiLLVMContext::get_constant(uint64 t);

#ifdef TI_PLATFORM_OSX
template llvm::Value *TaichiLLVMContext::get_constant(unsigned long t);
#endif

auto make_slim_libdevice = [](const std::vector<std::string> &args) {
  TI_ASSERT_INFO(args.size() == 1,
                 "Usage: ti task make_slim_libdevice [libdevice.X.bc file]");

  auto ctx = std::make_unique<llvm::LLVMContext>();
  auto libdevice_module = module_from_bitcode_file(args[0], ctx.get());

  remove_useless_cuda_libdevice_functions(libdevice_module.get());

  std::error_code ec;
  auto output_fn = "slim_" + args[0];
  llvm::raw_fd_ostream os(output_fn, ec, llvm::sys::fs::F_None);
  llvm::WriteBitcodeToFile(*libdevice_module, os);
  os.flush();
  TI_INFO("Slimmed libdevice written to {}", output_fn);
};

void TaichiLLVMContext::update_runtime_jit_module(
    std::unique_ptr<llvm::Module> module) {
  if (arch_ == Arch::cuda) {
    for (auto &f : *module) {
      bool is_kernel = false;
      const std::string func_name = f.getName();
      if (starts_with(func_name, "runtime_")) {
        mark_function_as_cuda_kernel(&f);
        is_kernel = true;
      }

      if (!is_kernel && !f.isDeclaration())
        // set declaration-only functions as internal linking to avoid
        // duplicated symbols and to remove external symbol dependencies such as
        // std::sin
        f.setLinkage(llvm::Function::PrivateLinkage);
    }
  }

  eliminate_unused_functions(module.get(), [](std::string func_name) {
    return starts_with(func_name, "runtime_") ||
           starts_with(func_name, "LLVMRuntime_");
  });
  runtime_jit_module = add_module(std::move(module));
}

void TaichiLLVMContext::delete_functions_of_snode_tree(int id) {
  if (!snode_tree_funcs_.count(id)) {
    return;
  }
  llvm::Module *module = get_this_thread_struct_module();
  for (auto str : snode_tree_funcs_[id]) {
    auto *func = module->getFunction(str);
    func->eraseFromParent();
  }
  snode_tree_funcs_.erase(id);
}

void TaichiLLVMContext::add_function_to_snode_tree(int id, std::string func) {
  snode_tree_funcs_[id].push_back(func);
}

TI_REGISTER_TASK(make_slim_libdevice);

}  // namespace lang
}  // namespace taichi
