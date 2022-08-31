#include "gtest/gtest.h"

#include "taichi/common/platform_macros.h"
#include "taichi/common/cleanup.h"

#ifdef TI_WITH_LLVM

#if defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_WINDOWS)
#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#error "Missing the <filesystem> header."
#endif  //  __has_include(<filesystem>)

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"

#include "taichi/rhi/arch.h"
#include "taichi/runtime/llvm/llvm_context.h"
#include "taichi/runtime/llvm/llvm_offline_cache.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/program/compile_config.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {
namespace {

constexpr char kKernelName[] = "foo";
constexpr char kTaskName[] = "my_add";
constexpr int kBlockDim = 1;
constexpr int kGridDim = 1;

using Format = LlvmOfflineCache::Format;

class LlvmOfflineCacheTest : public testing::TestWithParam<Format> {
 protected:
  void SetUp() override {
    const auto arch = host_arch();
    config_.packed = false;
    config_.print_kernel_llvm_ir = false;
    prog_ = std::make_unique<Program>(arch);
    auto *llvm_prog_ = get_llvm_program(prog_.get());
    tlctx_ = llvm_prog_->get_llvm_context(arch);
  }

  static std::unique_ptr<llvm::Module> make_module(
      llvm::LLVMContext &llvm_ctx) {
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

    llvm::verifyFunction(*func);
    return mod;
  }

  CompileConfig config_;
  // Program is *absolutely unnecessary* in this test. However, it is by far the
  // easiest approach in Taichi to use LLVM infra (e.g. JIT session).
  std::unique_ptr<Program> prog_{nullptr};
  TaichiLLVMContext *tlctx_{nullptr};
};

TEST_P(LlvmOfflineCacheTest, ReadWrite) {
  const auto llvm_fmt = GetParam();
  fs::path tmp_dir{fs::temp_directory_path() /= std::tmpnam(nullptr)};
  auto cleanup = make_cleanup([tmp_dir]() { fs::remove_all(tmp_dir); });
  const auto tmp_dir_str{tmp_dir.u8string()};
  const bool dir_ok = fs::create_directories(tmp_dir);
  ASSERT_TRUE(dir_ok);
  const std::vector<LlvmLaunchArgInfo> arg_infos = {
      LlvmLaunchArgInfo{/*is_array=*/false},
      LlvmLaunchArgInfo{/*is_array=*/true},
  };
  {
    auto llvm_ctx = std::make_unique<llvm::LLVMContext>();

    LlvmOfflineCacheFileWriter writer;
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
    LLVMCompiledData data;
    data.tasks = tasks;
    data.module = make_module(*llvm_ctx);
    kcache.compiled_data_list.push_back(std::move(data));
    kcache.args = arg_infos;
    writer.add_kernel_cache(kKernelName, std::move(kcache));
    writer.set_no_mangle();
    writer.dump(tmp_dir_str, llvm_fmt);
  }

  auto *llvm_ctx = tlctx_->get_this_thread_context();
  auto reader = LlvmOfflineCacheFileReader::make(tmp_dir_str, llvm_fmt);
  {
    LlvmOfflineCache::KernelCacheData kcache;
    const bool ok = reader->get_kernel_cache(kcache, kKernelName, *llvm_ctx);
    ASSERT_TRUE(ok);
    EXPECT_EQ(kcache.kernel_key, kKernelName);
    EXPECT_EQ(kcache.compiled_data_list[0].tasks.size(), 1);
    const auto &task0 = kcache.compiled_data_list[0].tasks.front();
    EXPECT_EQ(task0.name, kTaskName);

    ASSERT_NE(kcache.compiled_data_list[0].module, nullptr);
    kcache.compiled_data_list[0].module->dump();
    tlctx_->add_module(std::move(kcache.compiled_data_list[0].module));
    using FuncType = int (*)(int, int);
    FuncType my_add = (FuncType)tlctx_->lookup_function_pointer(kTaskName);
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
}  // namespace lang
}  // namespace taichi

#endif  // #if defined(TI_PLATFORM_LINUX) || defined(TI_PLATFORM_WINDOWS)
#endif  // #ifdef TI_WITH_LLVM
