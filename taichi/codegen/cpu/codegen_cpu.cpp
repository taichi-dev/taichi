// taichi/codegen/cpu/codegen_cpu.cpp

#include "taichi/codegen/cpu/codegen_cpu.h"

#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/util/file_sequence_writer.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"

// === CHANGED SECTION: HEADER INCLUDES ===
// #include "llvm/Support/Host.h" // Obsolete: This header has been removed.
#include "llvm/TargetParser/Host.h" // Replacement for Host.h to get sys::getHostCPUName()
#include "llvm/MC/TargetRegistry.h"
// #include "llvm/Transforms/IPO.h" // Obsolete: Part of the Legacy Pass Manager
// #include "llvm/Transforms/IPO/PassManagerBuilder.h" // Obsolete: Part of the Legacy Pass Manager
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
// New includes for the New Pass Manager (NPM)
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Transforms/Scalar/IndVarSimplify.h"
#include "llvm/Transforms/Scalar/LoopStrengthReduce.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/IPO/FunctionAttrs.h"
#include "llvm/Transforms/IPO/InferFunctionAttrs.h"
// === END OF CHANGED SECTION ===


namespace taichi::lang {

namespace {

class TaskCodeGenCPU : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenCPU(int id,
                 const CompileConfig &config,
                 TaichiLLVMContext &tlctx,
                 const Kernel *kernel,
                 IRNode *ir)
      : TaskCodeGenLLVM(id, config, tlctx, kernel, ir, nullptr) {
    TI_AUTO_PROF
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    int step = 1;

    // In parallel for-loops reversing the order doesn't make sense.
    // However, we may need to support serial offloaded range for's in the
    // future, so it still makes sense to reverse the order here.
    if (stmt->reversed) {
      step = -1;
    }

    auto *tls_prologue = create_xlogue(stmt->tls_prologue);

    // The loop body
    llvm::Function *body;
    {
      // === CHANGED SECTION: LLVM API CALL ===
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           // The call to `llvm::Type::getInt8PtrTy(*llvm_context)` was replaced.
           // `getInt8PtrTy` was a convenience function for getting an i8* type,
           // which has been removed. The modern equivalent for a generic pointer
           // is `getPointerTy()`, which creates an opaque `ptr` type.
           llvm::Type::getPointerTy(),
           tlctx->get_data_type<int>()});
      // === END OF CHANGED SECTION ===

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    llvm::Value *epilogue = create_xlogue(stmt->tls_epilogue);

    auto [begin, end] = get_range_for_bounds(stmt);

    call("cpu_parallel_range_for", get_arg(0),
         tlctx->get_constant(stmt->num_cpu_threads), begin, end,
         tlctx->get_constant(step), tlctx->get_constant(stmt->block_dim),
         tls_prologue, body, epilogue, tlctx->get_constant(stmt->tls_size));
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    auto *tls_prologue = create_mesh_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      // === CHANGED SECTION: LLVM API CALL ===
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           // Same change as in `create_offload_range_for`.
           // The call to `llvm::Type::getInt8PtrTy(*llvm_context)` was replaced
           // with the modern `llvm::Type::getPointerTy()`.
           llvm::Type::getPointerTy(),
           tlctx->get_data_type<int>()});
      // === END OF CHANGED SECTION ===

      for (int i = 0; i < stmt->mesh_prologue->size(); i++) {
        auto &s = stmt->mesh_prologue->statements[i];
        s->accept(this);
      }

      if (stmt->bls_prologue) {
        stmt->bls_prologue->accept(this);
      }

      auto loop_test_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_test", func);
      auto loop_body_bb =
          llvm::BasicBlock::Create(*llvm_context, "loop_body", func);
      auto func_exit =
          llvm::BasicBlock::Create(*llvm_context, "func_exit", func);
      auto loop_index =
          create_entry_block_alloca(llvm::Type::getInt32Ty(*llvm_context));
      builder->CreateStore(tlctx->get_constant(0), loop_index);
      builder->CreateBr(loop_test_bb);

      {
        builder->SetInsertPoint(loop_test_bb);
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
        auto cond = builder->CreateICmp(
            llvm::CmpInst::Predicate::ICMP_SLT, loop_index_load,
            llvm_val[stmt->owned_num_local.find(stmt->major_from_type)
                         ->second]);
        builder->CreateCondBr(cond, loop_body_bb, func_exit);
      }

      {
        builder->SetInsertPoint(loop_body_bb);
        loop_vars_llvm[stmt].push_back(loop_index);
        for (int i = 0; i < stmt->body->size(); i++) {
          auto &s = stmt->body->statements[i];
          s->accept(this);
        }
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
        builder->CreateStore(
            builder->CreateAdd(loop_index_load, tlctx->get_constant(1)),
            loop_index);
        builder->CreateBr(loop_test_bb);
        builder->SetInsertPoint(func_exit);
      }

      if (stmt->bls_epilogue) {
        stmt->bls_epilogue->accept(this);
      }

      body = guard.body;
    }

    llvm::Value *epilogue = create_mesh_xlogue(stmt->tls_epilogue);

    call("cpu_parallel_mesh_for", get_arg(0),
         tlctx->get_constant(stmt->num_cpu_threads),
         tlctx->get_constant(stmt->mesh->num_patches),
         tlctx->get_constant(stmt->block_dim), tls_prologue, body, epilogue,
         tlctx->get_constant(stmt->tls_size));
  }

  void create_bls_buffer(OffloadedStmt *stmt) {
    auto type = llvm::ArrayType::get(llvm::Type::getInt8Ty(*llvm_context),
                                     stmt->bls_size);
    bls_buffer = new llvm::GlobalVariable(
        *module, type, false, llvm::GlobalValue::ExternalLinkage, nullptr,
        "bls_buffer", nullptr, llvm::GlobalVariable::LocalExecTLSModel, 0);
    /* module->getOrInsertGlobal("bls_buffer", type);
    bls_buffer = module->getNamedGlobal("bls_buffer");
    bls_buffer->setAlignment(llvm::MaybeAlign(8));*/ // TODO(changyu): Fix JIT session error: Symbols not found: [ __emutls_get_address ] in python 3.10

    // initialize the variable with an undef value to ensure it is added to the
    // symbol table
    bls_buffer->setInitializer(llvm::UndefValue::get(type));
  }

  void visit(OffloadedStmt *stmt) override {
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (compile_config.kernel_profiler && arch_is_cpu(compile_config.arch)) {
      call("LLVMRuntime_profiler_start", get_runtime(),
           builder->CreateGlobalStringPtr(offloaded_task_name));
    }
    if (stmt->task_type == Type::serial) {
      stmt->body->accept(this);
    } else if (stmt->task_type == Type::range_for) {
      create_offload_range_for(stmt);
    } else if (stmt->task_type == Type::mesh_for) {
      create_offload_mesh_for(stmt);
    } else if (stmt->task_type == Type::struct_for) {
      stmt->block_dim = std::min(stmt->snode->parent->max_num_elements(),
                                 (int64)stmt->block_dim);
      create_offload_struct_for(stmt);
    } else if (stmt->task_type == Type::listgen) {
      emit_list_gen(stmt);
    } else if (stmt->task_type == Type::gc) {
      emit_gc(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
    if (compile_config.kernel_profiler && arch_is_cpu(compile_config.arch)) {
      llvm::IRBuilderBase::InsertPointGuard guard(*builder);
      builder->SetInsertPoint(final_block);
      call("LLVMRuntime_profiler_stop", get_runtime());
    }
    finalize_offloaded_task_function();
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
    current_offload = nullptr;
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    if (stmt->type == ExternalFuncCallStmt::BITCODE) {
      TaskCodeGenLLVM::visit_call_bitcode(stmt);
    } else if (stmt->type == ExternalFuncCallStmt::SHARED_OBJECT) {
      TaskCodeGenLLVM::visit_call_shared_object(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

 private:
  std::tuple<llvm::Value *, llvm::Value *> get_spmd_info() override {
    auto thread_idx = tlctx->get_constant(0);
    auto block_dim = tlctx->get_constant(1);
    return std::make_tuple(thread_idx, block_dim);
  }
};

static llvm::Triple get_host_target_triple() {
  auto expected_jtmb = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!expected_jtmb) {
    TI_ERROR("LLVM TargetMachineBuilder has failed.");
  }
  return expected_jtmb->getTargetTriple();
}

}  // namespace

#ifdef TI_WITH_LLVM
LLVMCompiledTask KernelCodeGenCPU::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenCPU gen(task_codegen_id, config, get_taichi_llvm_context(), kernel,
                     block);
  return gen.run_compilation();
}


// === CHANGED SECTION: ENTIRE FUNCTION REWRITTEN ===
// The `optimize_module` function has been completely rewritten to use the
// New Pass Manager (NPM) instead of the removed Legacy Pass Manager (LPM).
void KernelCodeGenCPU::optimize_module(llvm::Module *module) {
  TI_AUTO_PROF
  
  const auto &compile_config = get_compile_config();
  auto triple = get_host_target_triple();
  module->setTargetTriple(triple.str());

  std::string err_str;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.str(), err_str);
  TI_ERROR_UNLESS(target, err_str);

  llvm::TargetOptions options;
  if (compile_config.fast_math) {
    options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
    options.UnsafeFPMath = true;
    options.NoInfsFPMath = true;
    options.NoNaNsFPMath = true;
  }
  options.HonorSignDependentRoundingFPMathOption = false;
  options.NoZerosInBSS = false;
  options.GuaranteedTailCallOpt = false;

  llvm::StringRef mcpu = llvm::sys::getHostCPUName();
  std::unique_ptr<llvm::TargetMachine> target_machine(
      target->createTargetMachine(triple.str(), mcpu.str(), "", options,
                                  llvm::Reloc::PIC_, llvm::CodeModel::Small,
                                  llvm::CodeGenOpt::Aggressive));

  TI_ERROR_UNLESS(target_machine, "Could not allocate target machine!");

  module->setDataLayout(target_machine->createDataLayout());

  // === New Pass Manager Setup ===
  // 1. Create the analysis managers.
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;

  // 2. Create the PassBuilder. It's the main entry point for the NPM.
  llvm::PassBuilder PB(target_machine.get());

  // 3. Register all the standard analyses with the managers.
  FAM.registerPass([&] { return target_machine->getTargetIRAnalysis(); });
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // 4. Create the main pass manager.
  llvm::ModulePassManager MPM;

  // 5. Build the default optimization pipeline for a given level (e.g., O3).
  // This single line replaces the old `PassManagerBuilder` and `populate...` calls.
  llvm::PassBuilder::OptimizationLevel opt_level = llvm::PassBuilder::OptimizationLevel::O3;
  MPM = PB.buildPerModuleDefaultPipeline(opt_level);

  // 6. Add the custom passes that Taichi used in the old code.
  // Since they are FunctionPasses, they must be wrapped in an adaptor to be
  // added to a ModulePassManager.
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::LoopStrengthReducePass()));
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::IndVarSimplifyPass()));
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::SeparateConstOffsetFromGEPPass(false)));
  MPM.addPass(llvm::createModuleToFunctionPassAdaptor(llvm::EarlyCSEPass(true)));

  llvm::raw_svector_ostream *asm_ostream_ptr = nullptr;
  llvm::SmallString<0> asm_buffer;

  if (compile_config.print_kernel_asm) {
    // To emit assembly with NPM, you add a pass to the pipeline that writes to a stream.
    llvm::raw_svector_ostream &asm_stream = asm_buffer;
    asm_ostream_ptr = &asm_stream;
    asm_ostream_ptr->SetUnbuffered();
    if (auto err = target_machine->addPassesToEmitFile(MPM, *asm_ostream_ptr, nullptr, llvm::CGFT_AssemblyFile)) {
        TI_ERROR("Failed to addPassesToEmitFile");
    }
  }
  {
    TI_PROFILER("llvm_module_pass");
    MPM.run(*module, MAM);
  }

  // After MPM.run(), the asm_buffer will be populated if print_kernel_asm was true.
  if (compile_config.print_kernel_asm) {
    static FileSequenceWriter writer(
        "taichi_kernel_cpu_llvm_ir_optimized_asm_{:04d}.s",
        "optimized assembly code (CPU)");
    writer.write(std::string(asm_buffer.str()));
  }

  if (compile_config.print_kernel_llvm_ir_optimized) {
    if (false) {
      TI_INFO("Functions with > 100 instructions in optimized LLVM IR:");
      TaichiLLVMContext::print_huge_functions(module);
    }
    static FileSequenceWriter writer(
        "taichi_kernel_cpu_llvm_ir_optimized_{:04d}.ll",
        "optimized LLVM IR (CPU)");
    writer.write(module);
  }
}


#endif  // TI_WITH_LLVM
}  // namespace taichi::lang