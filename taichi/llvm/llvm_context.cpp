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
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Bitcode/BitcodeWriter.h"

#include "taichi/lang_util.h"
#include "taichi/jit/jit_session.h"
#include "taichi/common/task.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

TaichiLLVMContext::TaichiLLVMContext(Arch arch) : arch(arch) {
  TI_TRACE("Creating Taichi llvm context for arch: {}", arch_name(arch));
  main_thread_id = std::this_thread::get_id();
  main_thread_data = get_this_thread_data();
  llvm::remove_fatal_error_handler();
  llvm::install_fatal_error_handler(
      [](void *user_data, const std::string &reason, bool gen_crash_diag) {
        TI_ERROR("LLVM Fatal Error: {}", reason);
      },
      nullptr);

  if (arch_is_cpu(arch)) {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
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
  jit = JITSession::create(arch);
  TI_TRACE("Taichi llvm context created.");
}

llvm::Type *TaichiLLVMContext::get_data_type(DataType dt) {
  auto ctx = get_this_thread_context();
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
  } else if (dt == DataType::u8) {
    return llvm::Type::getInt8Ty(*ctx);
  } else if (dt == DataType::u16) {
    return llvm::Type::getInt16Ty(*ctx);
  } else if (dt == DataType::u32) {
    return llvm::Type::getInt32Ty(*ctx);
  } else if (dt == DataType::u64) {
    return llvm::Type::getInt64Ty(*ctx);
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
  for (auto cmd : commands) {
    TI_WARN("Potential command {}", cmd);
  }
  TI_ERROR("None command found.");
}

std::string get_runtime_fn(Arch arch) {
  return fmt::format("runtime_{}.bc", arch_name(arch));
}

std::string get_runtime_src_dir() {
  return get_repo_dir() + "/taichi/runtime/llvm/";
}

std::string get_runtime_dir() {
  if (runtime_tmp_dir.size() == 0)
    return get_runtime_src_dir();
  else
    return runtime_tmp_dir + "/runtime/";
}

void compile_runtime_bitcode(Arch arch) {
  if (is_release())
    return;
  TI_AUTO_PROF;
  static std::set<int> runtime_compiled;
  if (runtime_compiled.find((int)arch) == runtime_compiled.end()) {
    auto clang =
        find_existing_command({"clang-7", "clang-8", "clang-9", "clang"});
    TI_ASSERT(command_exist("llvm-as"));
    TI_TRACE("Compiling runtime module bitcode...");
    auto runtime_src_folder = get_runtime_src_dir();
    auto runtime_folder = get_runtime_dir();
    std::string macro = fmt::format(" -D ARCH_{} ", arch_name(arch));
    auto cmd = fmt::format(
        "{} -S {}runtime.cpp -o {}runtime.ll -fno-exceptions "
        "-emit-llvm -std=c++17 {} -I {}",
        clang, runtime_src_folder, runtime_folder, macro, get_repo_dir());
    int ret = std::system(cmd.c_str());
    if (ret) {
      TI_ERROR("LLVMRuntime compilation failed.");
    }
    cmd = fmt::format("llvm-as {}runtime.ll -o {}{}", runtime_folder,
                      runtime_folder, get_runtime_fn(arch));
    std::system(cmd.c_str());
    TI_TRACE("runtime module bitcode compiled.");
    runtime_compiled.insert((int)arch);
  }
}

void compile_runtimes() {
  compile_runtime_bitcode(host_arch());
#if defined(TI_WITH_CUDA)
  compile_runtime_bitcode(Arch::cuda);
#endif
}

std::string libdevice_path() {
  std::string folder;
  if (is_release()) {
    folder = compiled_lib_dir;
  } else {
    folder = fmt::format("{}/external/cuda_libdevice", get_repo_dir());
  }
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
    std::lock_guard<std::mutex> _(mut);
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

std::unique_ptr<llvm::Module> module_from_bitcode_file(std::string bitcode_path,
                                                       llvm::LLVMContext *ctx) {
  TI_AUTO_PROF
  std::ifstream ifs(bitcode_path, std::ios::binary);
  TI_ERROR_IF(!ifs, "Bitcode file ({}) not found.", bitcode_path);
  std::string bitcode(std::istreambuf_iterator<char>(ifs),
                      (std::istreambuf_iterator<char>()));
  auto runtime =
      parseBitcodeFile(llvm::MemoryBufferRef(bitcode, "runtime_bitcode"), *ctx);
  if (!runtime) {
    auto error = runtime.takeError();
    TI_WARN("Bitcode loading error message:");
    llvm::errs() << error << "\n";
    TI_ERROR("Bitcode {} load failure.", bitcode_path);
  }

  for (auto &f : *(runtime.get()))
    TaichiLLVMContext::force_inline(&f);

  bool module_broken = llvm::verifyModule(*runtime.get(), &llvm::errs());
  TI_ERROR_IF(module_broken, "Module broken");
  return std::move(runtime.get());
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
  TI_ASSERT(std::this_thread::get_id() == main_thread_id);
  auto data = get_this_thread_data();
  auto ctx = get_this_thread_context();
  if (!data->runtime_module) {
    if (is_release()) {
      data->runtime_module = module_from_bitcode_file(
          fmt::format("{}/{}", compiled_lib_dir, get_runtime_fn(arch)), ctx);
    } else {
      compile_runtime_bitcode(arch);
      data->runtime_module = module_from_bitcode_file(
          fmt::format("{}/{}", get_runtime_dir(), get_runtime_fn(arch)), ctx);
    }
    if (arch == Arch::cuda) {
      auto &runtime_module = data->runtime_module;
      runtime_module->setTargetTriple("nvptx64-nvidia-cuda");

      auto patch_intrinsic = [&](std::string name, Intrinsic::ID intrin,
                                 bool ret = true,
                                 std::vector<Type *> types = {}) {
        auto func = runtime_module->getFunction(name);
        func->deleteBody();
        auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
        IRBuilder<> builder(*ctx);
        builder.SetInsertPoint(bb);
        std::vector<llvm::Value *> args;
        for (auto &arg : func->args())
          args.push_back(&arg);
        if (ret) {
          builder.CreateRet(builder.CreateIntrinsic(intrin, types, args));
        } else {
          builder.CreateIntrinsic(intrin, types, args);
          builder.CreateRetVoid();
        }
        TaichiLLVMContext::force_inline(func);
      };

      auto patch_atomic_add = [&](std::string name,
                                  llvm::AtomicRMWInst::BinOp op) {
        auto func = runtime_module->getFunction(name);
        func->deleteBody();
        auto bb = llvm::BasicBlock::Create(*ctx, "entry", func);
        IRBuilder<> builder(*ctx);
        builder.SetInsertPoint(bb);
        std::vector<llvm::Value *> args;
        for (auto &arg : func->args())
          args.push_back(&arg);
        builder.CreateRet(builder.CreateAtomicRMW(
            op, args[0], args[1],
            llvm::AtomicOrdering::SequentiallyConsistent));
        TaichiLLVMContext::force_inline(func);
      };

      patch_intrinsic("thread_idx", Intrinsic::nvvm_read_ptx_sreg_tid_x);
      patch_intrinsic("block_idx", Intrinsic::nvvm_read_ptx_sreg_ctaid_x);
      patch_intrinsic("block_dim", Intrinsic::nvvm_read_ptx_sreg_ntid_x);
      patch_intrinsic("grid_dim", Intrinsic::nvvm_read_ptx_sreg_nctaid_x);
      patch_intrinsic("block_barrier", Intrinsic::nvvm_barrier0, false);
      patch_intrinsic("block_memfence", Intrinsic::nvvm_membar_cta, false);
      patch_intrinsic("grid_memfence", Intrinsic::nvvm_membar_gl, false);
      patch_intrinsic("system_memfence", Intrinsic::nvvm_membar_sys, false);

      patch_atomic_add("atomic_add_i32", llvm::AtomicRMWInst::Add);

      patch_atomic_add("atomic_add_i64", llvm::AtomicRMWInst::Add);

#if LLVM_VERSION_MAJOR >= 10
      patch_atomic_add("atomic_add_f32", llvm::AtomicRMWInst::FAdd);

      patch_atomic_add("atomic_add_f64", llvm::AtomicRMWInst::FAdd);
#else
      patch_intrinsic(
          "atomic_add_f32", Intrinsic::nvvm_atomic_load_add_f32, true,
          {llvm::PointerType::get(get_data_type(DataType::f32), 0)});

      patch_intrinsic(
          "atomic_add_f64", Intrinsic::nvvm_atomic_load_add_f64, true,
          {llvm::PointerType::get(get_data_type(DataType::f64), 0)});
#endif

      // patch_intrinsic("sync_warp", Intrinsic::nvvm_bar_warp_sync, false);
      // patch_intrinsic("warp_ballot", Intrinsic::nvvm_vote_ballot, false);
      // patch_intrinsic("warp_active_mask", Intrinsic::nvvm_membar_cta, false);
      patch_intrinsic("block_memfence", Intrinsic::nvvm_membar_cta, false);

      link_module_with_cuda_libdevice(data->runtime_module);

      // To prevent potential symbol name conflicts, we use "cuda_vprintf"
      // instead of "vprintf" in llvm/runtime.cpp. Now we change it back for
      // linking
      for (auto &f : *runtime_module) {
        if (f.getName() == "cuda_vprintf") {
          f.setName("vprintf");
        }
      }

      // runtime_module->print(llvm::errs(), nullptr);
    }
  }

  std::unique_ptr<llvm::Module> cloned;
  {
    TI_PROFILER("clone module");
    cloned = llvm::CloneModule(*data->runtime_module);
  }

  TI_ASSERT(cloned != nullptr);

  return cloned;
}

void TaichiLLVMContext::link_module_with_cuda_libdevice(
    std::unique_ptr<llvm::Module> &module) {
  TI_AUTO_PROF
  TI_ASSERT(arch == Arch::cuda);

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
      func->setLinkage(Function::InternalLinkage);
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
  data->struct_module = llvm::CloneModule(*module);
  if (!arch_is_cpu(arch)) {
    for (auto &f : *data->struct_module) {
      bool is_kernel = false;
      if (arch == Arch::cuda) {
        std::string func_name = f.getName();
        if (starts_with(func_name, "runtime_")) {
          mark_function_as_cuda_kernel(&f);
          is_kernel = true;
        }
      }

      if (!is_kernel && !f.isDeclaration())
        // set declaration-only functions as internal linking to avoid
        // duplicated symbols and to remove external symbol dependencies such as
        // std::sin
        f.setLinkage(llvm::Function::PrivateLinkage);
    }
  }

  auto runtime_module = clone_struct_module();
  eliminate_unused_functions(runtime_module.get(), [](std::string func_name) {
    return starts_with(func_name, "runtime_") ||
           starts_with(func_name, "LLVMRuntime_");
  });
  runtime_jit_module = add_module(std::move(runtime_module));
}

template <typename T>
llvm::Value *TaichiLLVMContext::get_constant(DataType dt, T t) {
  auto ctx = get_this_thread_context();
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
    TI_NOT_IMPLEMENTED
  }
}

template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, int32 t);
template llvm::Value *TaichiLLVMContext::get_constant(DataType dt, int64 t);
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
  return jit->get_type_size(type);
}

void TaichiLLVMContext::force_inline(llvm::Function *f) {
  f->removeAttribute(AttributeList::FunctionIndex,
                     llvm::Attribute::OptimizeNone);
  f->removeAttribute(AttributeList::FunctionIndex, llvm::Attribute::NoInline);
  f->addAttribute(AttributeList::FunctionIndex, llvm::Attribute::AlwaysInline);
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

TaichiLLVMContext::~TaichiLLVMContext() {
}

llvm::DataLayout TaichiLLVMContext::get_data_layout() {
  return jit->get_data_layout();
}

JITModule *TaichiLLVMContext::add_module(std::unique_ptr<llvm::Module> module) {
  return jit->add_module(std::move(module));
}

void TaichiLLVMContext::mark_function_as_cuda_kernel(llvm::Function *func) {
  auto ctx = get_this_thread_context();
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

  // Mark kernel function as a CUDA __global__ function
  // Add the nvvm annotation that it is considered a kernel function.

  llvm::Metadata *md_args[] = {llvm::ValueAsMetadata::get(func),
                               MDString::get(*ctx, "kernel"),
                               llvm::ValueAsMetadata::get(get_constant(1))};

  MDNode *md_node = MDNode::get(*ctx, md_args);

  func->getParent()
      ->getOrInsertNamedMetadata("nvvm.annotations")
      ->addOperand(md_node);
}

void TaichiLLVMContext::eliminate_unused_functions(
    llvm::Module *module,
    std::function<bool(const std::string &)> export_indicator) {
  TI_AUTO_PROF
  using namespace llvm;
  legacy::PassManager manager;
  ModuleAnalysisManager ana;
  manager.add(createInternalizePass([&](const GlobalValue &val) -> bool {
    return export_indicator(val.getName());
  }));
  manager.add(createGlobalDCEPass());
  manager.run(*module);
}

TaichiLLVMContext::ThreadLocalData *TaichiLLVMContext::get_this_thread_data() {
  std::lock_guard<std::mutex> _(thread_map_mut);
  auto tid = std::this_thread::get_id();
  if (per_thread_data.find(tid) == per_thread_data.end()) {
    std::stringstream ss;
    ss << tid;
    TI_TRACE("Creating thread local data for thread {}", ss.str());
    per_thread_data[tid] = std::make_unique<ThreadLocalData>();
  }
  return per_thread_data[tid].get();
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
        main_thread_data->struct_module.get());
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
                 "Usage: ti run make_slim_libdevice [libdevice.X.bc file]");

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

TI_REGISTER_TASK(make_slim_libdevice);

TLANG_NAMESPACE_END
