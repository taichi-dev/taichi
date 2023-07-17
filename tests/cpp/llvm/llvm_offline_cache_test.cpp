#include "gtest/gtest.h"

#include "taichi/common/platform_macros.h"
#include "taichi/common/cleanup.h"
#include "taichi/common/filesystem.hpp"

#ifdef TI_WITH_LLVM
#if defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_WINDOWS)

namespace fs = std::filesystem;

#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"

#include "taichi/rhi/arch.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/program.h"

namespace taichi::lang {
namespace {

static llvm::Triple get_host_target_triple() {
  auto expected_jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!expected_jtmb) {
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  }
  return expected_jtmb->getTargetTriple();
}

static std::unique_ptr<llvm::TargetMachine> get_host_target_machine() {
  auto triple = get_host_target_triple();

  std::string err_str;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  llvm::TargetOptions options;
  options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  options.UnsafeFPMath = 1;
  options.NoInfsFPMath = 1;
  options.NoNaNsFPMath = 1;
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;

  llvm::StringRef mcpu = llvm::sys::getHostCPUName();
  std::unique_ptr<llvm::TargetMachine> target_machine(
      target->createTargetMachine(triple.str(), mcpu.str(), "", options,
                                  llvm::Reloc::PIC_, llvm::CodeModel::Small,
                                  llvm::CodeGenOpt::Aggressive));
  return target_machine;
}

constexpr char kKernelName[] = "foo";
constexpr char kTaskName[] = "my_add";
constexpr int kBlockDim = 1;
constexpr int kGridDim = 1;

using Format = LlvmOfflineCache::Format;

class LlvmOfflineCacheTest : public testing::TestWithParam<Format> {
 protected:
  void SetUp() override {
    const auto arch = host_arch();
    config_.print_kernel_llvm_ir = false;
    prog_ = std::make_unique<Program>(arch);
    auto *llvm_prog_ = get_llvm_program(prog_.get());
    tlctx_ = llvm_prog_->get_llvm_context();
    executor_ = llvm_prog_->get_runtime_executor();
  }

  static std::unique_ptr<llvm::Module> make_module(
      llvm::LLVMContext &llvm_ctx) {
    auto target_machine = get_host_target_machine();

    auto mod = std::make_unique<llvm::Module>("my_mod", llvm_ctx);
    auto builder = std::make_unique<llvm::IRBuilder<>>(llvm_ctx);
    auto *const int32_ty = llvm::Type::getInt32Ty(llvm_ctx);
    auto *const func_ty =
        llvm::FunctionType::get(int32_ty, {int32_ty, int32_ty},
                                /*isVarArg=*/false);
    auto *const func = llvm::Function::Create(
        func_ty, llvm::Function::ExternalLinkage, kTaskName, mod.get());
    std::vector<llvm::Value *> args;
    for (auto &a : func->args()) {
      args.push_back(&a);
    }
    auto *entry_block = llvm::BasicBlock::Create(llvm_ctx, "entry", func);
    builder->SetInsertPoint(entry_block);
    auto *ret_val = builder->CreateAdd(args[0], args[1], "add");
    builder->CreateRet(ret_val);

    mod->setDataLayout(target_machine->createDataLayout());

    llvm::verifyFunction(*func);
    return mod;
  }

  CompileConfig config_;
  // Program is *absolutely unnecessary* in this test. However, it is by far the
  // easiest approach in Taichi to use LLVM infra (e.g. JIT session).
  std::unique_ptr<Program> prog_{nullptr};
  TaichiLLVMContext *tlctx_{nullptr};
  LlvmRuntimeExecutor *executor_{nullptr};
};

TEST_P(LlvmOfflineCacheTest, ReadWrite) {
  const auto llvm_fmt = GetParam();
  fs::path tmp_dir{fs::temp_directory_path() /= std::tmpnam(nullptr)};
  auto cleanup = make_cleanup([tmp_dir]() { fs::remove_all(tmp_dir); });
  const auto tmp_dir_str{tmp_dir.u8string()};
  const bool dir_ok = fs::create_directories(tmp_dir);
  ASSERT_TRUE(dir_ok);
  const std::vector<std::pair<std::vector<int>, Callable::Parameter>>
      arg_infos = {
          {{0}, Callable::Parameter{DataType(PrimitiveType::i32), false}},
          {{1}, Callable::Parameter{DataType(PrimitiveType::i32), false}},
      };
  auto member1 = AbstractDictionaryMember{PrimitiveType::i32, "a"};
  auto member2 = AbstractDictionaryMember{PrimitiveType::i32, "b"};
  auto struct_type =
      TypeFactory::get_instance().get_struct_type({member1, member2});
  auto arg_type = tlctx_->get_struct_type_with_data_layout(
      struct_type->as<StructType>(), tlctx_->get_data_layout_string());
  {
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();

    llvm_ctx->setOpaquePointers(false);
    LlvmOfflineCache::KernelCacheData kcache;
    kcache.created_at = 1;
    kcache.last_used_at = 1;
    kcache.kernel_key = kKernelName;
    std::vector<OffloadedTask> tasks;
    OffloadedTask task;
    task.name = kTaskName;
    task.block_dim = kBlockDim;
    task.grid_dim = kGridDim;
    tasks.push_back(task);
    kcache.compiled_data.tasks = tasks;
    kcache.compiled_data.module = make_module(*llvm_ctx);
    kcache.args = arg_infos;
    kcache.args_type = arg_type.first;
    kcache.args_size = arg_type.second;
    LlvmOfflineCacheFileWriter writer1;
    writer1.add_kernel_cache(kKernelName, kcache.clone());
    writer1.set_no_mangle();
    writer1.dump(tmp_dir_str, llvm_fmt, /*merge_with_old=*/false);
    // Dump twice to verify the correctness of LlvmOfflineCacheFileWriter::dump
    LlvmOfflineCacheFileWriter writer2;
    writer2.add_kernel_cache(kKernelName, kcache.clone());
    writer2.set_no_mangle();
    writer2.dump(tmp_dir_str, llvm_fmt, /*merge_with_old=*/false);
  }

  auto *llvm_ctx = tlctx_->get_this_thread_context();
  auto reader = LlvmOfflineCacheFileReader::make(tmp_dir_str, llvm_fmt);
  {
    LlvmOfflineCache::KernelCacheData kcache;
    const bool ok = reader->get_kernel_cache(kcache, kKernelName, *llvm_ctx);
    ASSERT_TRUE(ok);
    EXPECT_EQ(kcache.kernel_key, kKernelName);
    EXPECT_EQ(kcache.compiled_data.tasks.size(), 1);
    const auto &task0 = kcache.compiled_data.tasks.front();
    EXPECT_EQ(task0.name, kTaskName);

    ASSERT_NE(kcache.compiled_data.module, nullptr);
    kcache.compiled_data.module->dump();
    auto jit_module =
        executor_->create_jit_module(std::move(kcache.compiled_data.module));
    using FuncType = int (*)(int, int);
    FuncType my_add = (FuncType)jit_module->lookup_function(kTaskName);
    const auto res = my_add(40, 2);
    EXPECT_EQ(res, 42);
  };
  {
    // Do it twice. No file IO this time.
    LlvmOfflineCache::KernelCacheData kcache;
    const bool ok = reader->get_kernel_cache(kcache, kKernelName, *llvm_ctx);
    ASSERT_TRUE(ok);
    const auto &actual_arg_infos = kcache.args;
    EXPECT_EQ(actual_arg_infos, arg_infos);
  };
}

INSTANTIATE_TEST_SUITE_P(Format,
                         LlvmOfflineCacheTest,
                         testing::Values(Format::LL, Format::BC));

}  // namespace
}  // namespace taichi::lang

#endif  // #if defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_WINDOWS)
#endif  // #ifdef TI_WITH_LLVM
