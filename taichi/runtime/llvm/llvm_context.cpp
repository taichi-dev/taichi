// A llvm backend helper

#include "taichi/runtime/llvm/llvm_context.h"

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
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#ifdef TI_WITH_AMDGPU
#include "llvm/IR/IntrinsicsAMDGPU.h"
#endif  // TI_WITH_AMDGPU
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/FileSystem.h"
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

#include "taichi/util/lang_util.h"
#include "taichi/jit/jit_session.h"
#include "taichi/common/task.h"
#include "taichi/util/environ_config.h"
#include "llvm_context.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/codegen/codegen_utils.h"

#include "taichi/runtime/llvm/llvm_context_pass.h"

#ifdef _WIN32
// Travis CI seems doesn't support <filesystem>...
#include <filesystem>
#else
#include <unistd.h>
#endif

#if defined(TI_WITH_CUDA)
#include "taichi/rhi/cuda/cuda_context.h"
#endif

#if defined(TI_WITH_AMDGPU)
#include "taichi/rhi/amdgpu/amdgpu_context.h"
#endif

namespace taichi::lang {

using namespace llvm;

TaichiLLVMContext::TaichiLLVMContext(const CompileConfig &config, Arch arch)
    : config_(config), arch_(arch) {
  TI_TRACE("Creating Taichi llvm context for arch: {}", arch_name(arch));
  main_thread_id_ = std::this_thread::get_id();
  main_thread_data_ = get_this_thread_data();
  llvm::remove_fatal_error_handler();
  llvm::install_fatal_error_handler(
      [](void *user_data, const char *reason, bool gen_crash_diag) {
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
  } else if (arch == Arch::dx12) {
    // FIXME: Must initialize these before initializing Arch::dx12
    // because it uses the jit of CPU right now.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
    // The dx target is used elsewhere, so we need to initialize it too.
#if defined(TI_WITH_DX12)
    LLVMInitializeDirectXTarget();
    LLVMInitializeDirectXTargetMC();
    LLVMInitializeDirectXTargetInfo();
    LLVMInitializeDirectXAsmPrinter();
#endif
  } else if (arch == Arch::amdgpu) {
#if defined(TI_WITH_AMDGPU)
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUAsmPrinter();
    LLVMInitializeAMDGPUAsmParser();
#else
    TI_NOT_IMPLEMENTED
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

  data_layout_ = TaichiLLVMContext::get_data_layout(arch);
  linking_context_data = std::make_unique<ThreadLocalData>(
      std::make_unique<llvm::orc::ThreadSafeContext>(
          std::make_unique<llvm::LLVMContext>()));
  linking_context_data->runtime_module = clone_module_to_context(
      get_this_thread_runtime_module(), linking_context_data->llvm_context);

  TI_TRACE("Taichi llvm context created.");
}

TaichiLLVMContext::~TaichiLLVMContext() {
}

llvm::Type *TaichiLLVMContext::get_data_type(DataType dt) {
  auto ctx = get_this_thread_context();
  if (dt->is_primitive(PrimitiveTypeID::i8) ||
      dt->is_primitive(PrimitiveTypeID::u8)) {
    return llvm::Type::getInt8Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::i16) ||
             dt->is_primitive(PrimitiveTypeID::u16)) {
    return llvm::Type::getInt16Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::i32) ||
             dt->is_primitive(PrimitiveTypeID::u32)) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::i64) ||
             dt->is_primitive(PrimitiveTypeID::u64)) {
    return llvm::Type::getInt64Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return llvm::Type::getInt1Ty(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::f32)) {
    return llvm::Type::getFloatTy(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::f64)) {
    return llvm::Type::getDoubleTy(*ctx);
  } else if (dt->is_primitive(PrimitiveTypeID::f16)) {
    return llvm::Type::getHalfTy(*ctx);
  } else if (const auto *tensor_type = dt->cast<TensorType>()) {
    auto element_type = get_data_type(tensor_type->get_element_type());
    auto num_elements = tensor_type->get_num_elements();
    // Return type is <element_type * num_elements> if real matrix is used,
    // otherwise [element_type * num_elements].
    if (codegen_vector_type(config_)) {
      return llvm::VectorType::get(element_type, num_elements,
                                   /*scalable=*/false);
    }
    return llvm::ArrayType::get(element_type, num_elements);
  } else if (const auto *struct_type = dt->cast<StructType>()) {
    std::vector<llvm::Type *> types;
    for (const auto &element : struct_type->elements()) {
      types.push_back(get_data_type(element.type));
    }
    return llvm::StructType::get(*ctx, types);
  } else if (const auto *argpack_type = dt->cast<ArgPackType>()) {
    std::vector<llvm::Type *> types;
    for (const auto &element : argpack_type->elements()) {
      types.push_back(get_data_type(element.type));
    }
    return llvm::StructType::get(*ctx, types);
  } else if (const auto *pointer_type = dt->cast<PointerType>()) {
    return llvm::PointerType::get(
        get_data_type(pointer_type->get_pointee_type()), 0);
  } else {
    TI_INFO(data_type_name(dt));
    TI_NOT_IMPLEMENTED;
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

// Note: runtime_module = init_module < struct_module

std::unique_ptr<llvm::Module> TaichiLLVMContext::clone_runtime_module() {
  TI_AUTO_PROF
  auto *mod = get_this_thread_runtime_module();

  std::unique_ptr<llvm::Module> cloned;
  {
    TI_PROFILER("clone module");
    cloned = llvm::CloneModule(*mod);
  }

  TI_ASSERT(cloned != nullptr);

  return cloned;
}

std::unique_ptr<llvm::Module> TaichiLLVMContext::module_from_file(
    const std::string &file) {
  auto ctx = get_this_thread_context();
  std::unique_ptr<llvm::Module> module = module_from_bitcode_file(
      fmt::format("{}/{}", runtime_lib_dir(), file), ctx);
  if (arch_ == Arch::cuda || arch_ == Arch::amdgpu) {
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
          op, args[0], args[1], llvm::MaybeAlign(0),
          llvm::AtomicOrdering::SequentiallyConsistent));
      TaichiLLVMContext::mark_inline(func);
    };

    patch_atomic_add("atomic_add_i32", llvm::AtomicRMWInst::Add);
    patch_atomic_add("atomic_add_i64", llvm::AtomicRMWInst::Add);
    patch_atomic_add("atomic_add_f64", llvm::AtomicRMWInst::FAdd);
    patch_atomic_add("atomic_add_f32", llvm::AtomicRMWInst::FAdd);

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

      patch_intrinsic("cuda_shfl_up_sync_i32",
                      Intrinsic::nvvm_shfl_sync_up_i32);
      patch_intrinsic("cuda_shfl_up_sync_f32",
                      Intrinsic::nvvm_shfl_sync_up_f32);

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

      patch_intrinsic("block_memfence", Intrinsic::nvvm_membar_cta, false);

      link_module_with_cuda_libdevice(module);

#ifdef TI_WITH_CUDA
      int cap = CUDAContext::get_instance().get_compute_capability();
      if (cap >= 60)
        link_module_with_custom_cuda_library(module);
#endif

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

#ifdef TI_WITH_AMDGPU
    auto patch_amdgpu_kernel_dim = [&](std::string name, llvm::Value *lhs) {
      std::string actual_name;
      if (name == "block_dim")
        actual_name = "__ockl_get_local_size";
      else if (name == "grid_dim")
        actual_name = "__ockl_get_num_groups";
      else
        TI_ERROR("Unknown patch function name");
      auto func = module->getFunction(name);
      auto actual_func = module->getFunction(actual_name);
      if (!func || !actual_func) {
        return;
      }
      func->deleteBody();
      auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
      IRBuilder<> builder(*ctx);
      builder.SetInsertPoint(bb);
      auto dim_ = builder.CreateCall(actual_func->getFunctionType(),
                                     actual_func, {lhs});
      auto ret_ = builder.CreateTrunc(dim_, llvm::Type::getInt32Ty(*ctx));
      builder.CreateRet(ret_);
      TaichiLLVMContext::mark_inline(func);
    };
#endif

    if (arch_ == Arch::amdgpu) {
      module->setTargetTriple("amdgcn-amd-amdhsa");
#ifdef TI_WITH_AMDGPU
      llvm::legacy::FunctionPassManager function_pass_manager(module.get());
      function_pass_manager.add(new AMDGPUConvertAllocaInstAddressSpacePass());
      function_pass_manager.doInitialization();
      for (auto func = module->begin(); func != module->end(); ++func) {
        function_pass_manager.run(*func);
      }
      function_pass_manager.doFinalization();
      patch_intrinsic("thread_idx", llvm::Intrinsic::amdgcn_workitem_id_x);
      patch_intrinsic("block_idx", llvm::Intrinsic::amdgcn_workgroup_id_x);
      patch_intrinsic("block_barrier", llvm::Intrinsic::amdgcn_s_barrier,
                      false);

      link_module_with_amdgpu_libdevice(module);
      patch_amdgpu_kernel_dim(
          "block_dim", llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 0));
      patch_amdgpu_kernel_dim(
          "grid_dim", llvm::ConstantInt::get(llvm::Type::getInt32Ty(*ctx), 0));
#endif
    }
  }

  return module;
}

void TaichiLLVMContext::link_module_with_custom_cuda_library(
    std::unique_ptr<llvm::Module> &module) {
  std::string cuda_library_path = get_custom_cuda_library_path();
  if (!cuda_library_path.empty()) {
    std::unique_ptr<llvm::Module> cuda_library_module =
        module_from_bitcode_file(
            fmt::format("{}/{}", runtime_lib_dir(),
                        "cuda_runtime-cuda-nvptx64-nvidia-cuda-sm_60.bc"),
            get_this_thread_context());

    module->setDataLayout(cuda_library_module->getDataLayout());
    bool failed =
        llvm::Linker::linkModules(*module, std::move(cuda_library_module));
    if (failed) {
      TI_ERROR("cuda_runtime.bc linking failure.");
    }
  }
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
      libdevice_function_names.push_back(f.getName().str());
    }
  }

  libdevice_module->setTargetTriple("nvptx64-nvidia-cuda");
  module->setDataLayout(libdevice_module->getDataLayout());

  bool failed = llvm::Linker::linkModules(*module, std::move(libdevice_module));
  if (failed) {
    TI_ERROR("CUDA libdevice linking failure.");
  }

  // Make sure all libdevice functions are linked
  for (auto func_name : libdevice_function_names) {
    auto func = module->getFunction(func_name);
    if (!func) {
      TI_INFO("Function {} not found", func_name);
    }
  }
}

void TaichiLLVMContext::link_module_with_amdgpu_libdevice(
    std::unique_ptr<llvm::Module> &module) {
  TI_ASSERT(arch_ == Arch::amdgpu);
#if defined(TI_WITH_AMDGPU)
  auto isa_version = AMDGPUContext::get_instance().get_mcpu().substr(3, 4);
  std::string libdevice_files[] = {"ocml.bc",
                                   "oclc_wavefrontsize64_off.bc",
                                   "ockl.bc",
                                   "oclc_abi_version_400.bc",
                                   "oclc_correctly_rounded_sqrt_off.bc",
                                   "oclc_daz_opt_off.bc",
                                   "oclc_finite_only_off.bc",
                                   "oclc_isa_version_" + isa_version + ".bc",
                                   "oclc_unsafe_math_off.bc",
                                   "opencl.bc"};

  for (auto &libdevice : libdevice_files) {
    std::string lib_dir = runtime_lib_dir() + "/";
    auto libdevice_module = module_from_bitcode_file(lib_dir + libdevice,
                                                     get_this_thread_context());

    if (libdevice == "ocml.bc")
      module->setDataLayout(libdevice_module->getDataLayout());

    std::vector<std::string> libdevice_func_names;
    for (auto &f : *libdevice_module) {
      if (!f.isDeclaration()) {
        libdevice_func_names.push_back(f.getName().str());
      }
    }

    for (auto &f : libdevice_module->functions()) {
      auto func_name = libdevice.substr(0, libdevice.length() - 3);
      if (starts_with(f.getName().lower(), "__" + func_name))
        f.setLinkage(llvm::Function::CommonLinkage);
    }

    bool failed =
        llvm::Linker::linkModules(*module, std::move(libdevice_module));
    if (failed) {
      TI_ERROR("AMDGPU libdevice linking failure.");
    }
  }
#endif
}

void TaichiLLVMContext::add_struct_module(std::unique_ptr<Module> module,
                                          int tree_id) {
  TI_AUTO_PROF;
  TI_ASSERT(std::this_thread::get_id() == main_thread_id_);
  auto this_thread_data = get_this_thread_data();
  TI_ASSERT(module);
  if (llvm::verifyModule(*module, &llvm::errs())) {
    module->print(llvm::errs(), nullptr);
    TI_ERROR("module broken");
  }

  linking_context_data->struct_modules[tree_id] =
      clone_module_to_context(module.get(), linking_context_data->llvm_context);

  for (auto &[id, data] : per_thread_data_) {
    if (id == std::this_thread::get_id()) {
      continue;
    }
    data->struct_modules[tree_id] =
        clone_module_to_context(module.get(), data->llvm_context);
  }

  this_thread_data->struct_modules[tree_id] = std::move(module);
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
  } else if (dt->is_primitive(PrimitiveTypeID::u1)) {
    return t ? llvm::ConstantInt::getTrue(*ctx)
             : llvm::ConstantInt::getFalse(*ctx);
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
    return t ? llvm::ConstantInt::getTrue(*ctx)
             : llvm::ConstantInt::getFalse(*ctx);
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
  f->removeFnAttr(llvm::Attribute::OptimizeNone);
  f->removeFnAttr(llvm::Attribute::NoInline);
  f->addFnAttr(llvm::Attribute::AlwaysInline);
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
  return data_layout_;
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

void TaichiLLVMContext::mark_function_as_amdgpu_kernel(llvm::Function *func) {
  func->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
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
    return export_indicator(val.getName().str());
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
    per_thread_data_[tid] = std::make_unique<ThreadLocalData>(
        std::make_unique<llvm::orc::ThreadSafeContext>(
            std::make_unique<llvm::LLVMContext>()));
  }
  return per_thread_data_[tid].get();
}

llvm::LLVMContext *TaichiLLVMContext::get_this_thread_context() {
  ThreadLocalData *data = get_this_thread_data();
  TI_ASSERT(data->llvm_context)
  return data->llvm_context;
}

llvm::orc::ThreadSafeContext *
TaichiLLVMContext::get_this_thread_thread_safe_context() {
  get_this_thread_context();  // make sure the context is created
  ThreadLocalData *data = get_this_thread_data();
  return data->thread_safe_llvm_context.get();
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
  llvm::raw_fd_ostream os(output_fn, ec, llvm::sys::fs::OF_None);
  llvm::WriteBitcodeToFile(*libdevice_module, os);
  os.flush();
  TI_INFO("Slimmed libdevice written to {}", output_fn);
};

void TaichiLLVMContext::init_runtime_module(llvm::Module *runtime_module) {
  if (config_.arch == Arch::cuda) {
    for (auto &f : *runtime_module) {
      bool is_kernel = false;
      const std::string func_name = f.getName().str();
      if (starts_with(func_name, "runtime_")) {
        mark_function_as_cuda_kernel(&f);
        is_kernel = true;
      }

      if (!is_kernel && !f.isDeclaration())
        // set declaration-only functions as internal linking to avoid
        // duplicated symbols and to remove external symbol dependencies such
        // as std::sin
        f.setLinkage(llvm::Function::PrivateLinkage);
    }
  }

  if (config_.arch == Arch::amdgpu) {
#ifdef TI_WITH_AMDGPU
    llvm::legacy::PassManager module_pass_manager;
    module_pass_manager.add(new AMDGPUConvertFuncParamAddressSpacePass());
    module_pass_manager.run(*runtime_module);
#endif
  }

  eliminate_unused_functions(runtime_module, [](std::string func_name) {
    return starts_with(func_name, "runtime_") ||
           starts_with(func_name, "LLVMRuntime_");
  });
}

void TaichiLLVMContext::delete_snode_tree(int id) {
  TI_ASSERT(linking_context_data->struct_modules.erase(id));
  for (auto &[thread_id, data] : per_thread_data_) {
    TI_ASSERT(data->struct_modules.erase(id));
  }
}

void TaichiLLVMContext::fetch_this_thread_struct_module() {
  ThreadLocalData *data = get_this_thread_data();
  if (data->struct_modules.empty()) {
    for (auto &[id, mod] : main_thread_data_->struct_modules) {
      data->struct_modules[id] = clone_module_to_this_thread_context(mod.get());
    }
  }
}

llvm::Function *TaichiLLVMContext::get_runtime_function(
    const std::string &name) {
  return get_this_thread_runtime_module()->getFunction(name);
}

llvm::Module *TaichiLLVMContext::get_this_thread_runtime_module() {
  TI_AUTO_PROF;
  auto data = get_this_thread_data();
  if (!data->runtime_module) {
    data->runtime_module = module_from_file(get_runtime_fn(arch_));
  }
  return data->runtime_module.get();
}

llvm::Function *TaichiLLVMContext::get_struct_function(const std::string &name,
                                                       int tree_id) {
  auto *data = get_this_thread_data();
  return data->struct_modules[tree_id]->getFunction(name);
}

llvm::Type *TaichiLLVMContext::get_runtime_type(const std::string &name) {
  auto ty = llvm::StructType::getTypeByName(
      get_this_thread_runtime_module()->getContext(), ("struct." + name));
  if (!ty) {
    TI_ERROR("LLVMRuntime type {} not found.", name);
  }
  return ty;
}
std::unique_ptr<llvm::Module> TaichiLLVMContext::new_module(
    std::string name,
    llvm::LLVMContext *context) {
  auto new_mod = std::make_unique<llvm::Module>(
      name, context ? *context : *get_this_thread_context());
  new_mod->setDataLayout(get_this_thread_runtime_module()->getDataLayout());
  return new_mod;
}

TaichiLLVMContext::ThreadLocalData::ThreadLocalData(
    std::unique_ptr<llvm::orc::ThreadSafeContext> ctx)
    : thread_safe_llvm_context(std::move(ctx)),
      llvm_context(thread_safe_llvm_context->getContext()) {
}

TaichiLLVMContext::ThreadLocalData::~ThreadLocalData() {
  runtime_module.reset();
  struct_modules.clear();
  thread_safe_llvm_context.reset();
}

LLVMCompiledKernel TaichiLLVMContext::link_compiled_tasks(
    std::vector<std::unique_ptr<LLVMCompiledTask>> data_list) {
  LLVMCompiledKernel linked;
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> tls_sizes;
  std::unordered_set<std::string> offloaded_names;
  auto mod = new_module("kernel", linking_context_data->llvm_context);
  llvm::Linker linker(*mod);
  for (auto &datum : data_list) {
    for (auto tree_id : datum->used_tree_ids) {
      used_tree_ids.insert(tree_id);
    }
    for (auto tls_size : datum->struct_for_tls_sizes) {
      tls_sizes.insert(tls_size);
    }
    for (auto &task : datum->tasks) {
      offloaded_names.insert(task.name);
      linked.tasks.push_back(std::move(task));
    }
    linker.linkInModule(clone_module_to_context(
        datum->module.get(), linking_context_data->llvm_context));
  }
  for (auto tree_id : used_tree_ids) {
    linker.linkInModule(
        llvm::CloneModule(*linking_context_data->struct_modules[tree_id]),
        llvm::Linker::LinkOnlyNeeded | llvm::Linker::OverrideFromSrc);
  }
  auto runtime_module =
      llvm::CloneModule(*linking_context_data->runtime_module);
  for (auto tls_size : tls_sizes) {
    add_struct_for_func(runtime_module.get(), tls_size);
  }
  linker.linkInModule(
      std::move(runtime_module),
      llvm::Linker::LinkOnlyNeeded | llvm::Linker::OverrideFromSrc);
  eliminate_unused_functions(mod.get(), [&](std::string func_name) -> bool {
    return offloaded_names.count(func_name);
  });
  linked.module = std::move(mod);
  return linked;
}

void TaichiLLVMContext::add_struct_for_func(llvm::Module *module,
                                            int tls_size) {
  // Note that on CUDA local array allocation must have a compile-time
  // constant size. Therefore, instead of passing in the tls_buffer_size
  // argument, we directly clone the "parallel_struct_for" function and
  // replace the "alignas(8) char tls_buffer[1]" statement with "alignas(8)
  // char tls_buffer[tls_buffer_size]" at compile time.
  auto func_name = get_struct_for_func_name(tls_size);
  if (module->getFunction(func_name)) {
    return;
  }
  llvm::legacy::PassManager module_pass_manager;
  if (config_.arch == Arch::amdgpu) {
#ifdef TI_WITH_AMDGPU
    module_pass_manager.add(
        new AMDGPUAddStructForFuncPass(func_name, tls_size));
    module_pass_manager.run(*module);
#else
    TI_NOT_IMPLEMENTED
#endif
  } else {
    module_pass_manager.add(new AddStructForFuncPass(func_name, tls_size));
    module_pass_manager.run(*module);
  }
}

std::string TaichiLLVMContext::get_struct_for_func_name(int tls_size) {
  return "parallel_struct_for_" + std::to_string(tls_size);
}

llvm::DataLayout TaichiLLVMContext::get_data_layout(Arch arch) {
  TI_ASSERT(arch_uses_llvm(arch));
  if (arch_is_cpu(arch)) {
    auto expected_jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!expected_jtmb)
      TI_ERROR("LLVM TargetMachineBuilder has failed.");
    auto jtmb = *expected_jtmb;
    auto expected_data_layout = jtmb.getDefaultDataLayoutForTarget();
    if (!expected_data_layout) {
      TI_ERROR(
          "LLVM TargetMachineBuilder has failed when getting data layout.");
    }
    return *expected_data_layout;
  } else if (arch == Arch::cuda) {
    return llvm::DataLayout(
        "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-"
        "f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:"
        "64");
  } else if (arch == Arch::amdgpu) {
    return llvm::DataLayout(
        "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:"
        "64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-"
        "v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7");
  } else if (arch == Arch::dx12) {
    // NOTE: Return the default data layout to avoid crash.
    return llvm::DataLayout("");
  } else {
    TI_NOT_IMPLEMENTED
  }
}

std::string TaichiLLVMContext::get_data_layout_string() {
  return get_data_layout().getStringRepresentation();
}

std::pair<const StructType *, size_t>
TaichiLLVMContext::get_struct_type_with_data_layout(const StructType *old_ty,
                                                    const std::string &layout) {
  auto *llvm_struct_type = llvm::cast<llvm::StructType>(get_data_type(old_ty));
  auto data_layout = llvm::DataLayout::parse(layout);
  TI_ASSERT(data_layout);
  size_t struct_size = data_layout->getTypeAllocSize(llvm_struct_type);
  if (old_ty->get_layout() == layout) {
    return {old_ty, struct_size};
  }
  std::vector<AbstractDictionaryMember> elements = old_ty->elements();
  for (auto &element : elements) {
    if (auto struct_type = element.type->cast<StructType>()) {
      element.type =
          get_struct_type_with_data_layout(struct_type, layout).first;
    }
  }
  auto struct_layout = data_layout->getStructLayout(llvm_struct_type);
  for (int i = 0; i < elements.size(); i++) {
    elements[i].offset = struct_layout->getElementOffset(i);
  }
  return {TypeFactory::get_instance()
              .get_struct_type(elements, layout)
              ->cast<StructType>(),
          struct_size};
}

TI_REGISTER_TASK(make_slim_libdevice);

}  // namespace taichi::lang
