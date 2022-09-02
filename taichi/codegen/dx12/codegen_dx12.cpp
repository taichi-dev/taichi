#include "taichi/codegen/dx12/codegen_dx12.h"

#include "taichi/rhi/dx12/dx12_api.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/util/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/statistics.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/analysis/offline_cache_util.h"
TLANG_NAMESPACE_BEGIN

namespace {

class TaskCodeGenLLVMDX12 : public TaskCodeGenLLVM {
 public:
  using IRVisitor::visit;

  TaskCodeGenLLVMDX12(Kernel *kernel, IRNode *ir)
      : TaskCodeGenLLVM(kernel, ir, nullptr) {
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
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::Type::getInt8PtrTy(*llvm_context),
           tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    llvm::Value *epilogue = create_xlogue(stmt->tls_epilogue);

    auto [begin, end] = get_range_for_bounds(stmt);

    // adaptive block_dim
    if (prog->config.cpu_block_dim_adaptive) {
      int num_items = (stmt->end_value - stmt->begin_value) / std::abs(step);
      int num_threads = stmt->num_cpu_threads;
      int items_per_thread = std::max(1, num_items / (num_threads * 32));
      // keep each task has at least 512 items to amortize scheduler overhead
      // also saturate the value to 1024 for better load balancing
      stmt->block_dim = std::min(1024, std::max(512, items_per_thread));
    }

    create_call(
        "gpu_parallel_range_for",
        {get_arg(0), tlctx->get_constant(stmt->num_cpu_threads), begin, end,
         tlctx->get_constant(step), tlctx->get_constant(stmt->block_dim),
         tls_prologue, body, epilogue, tlctx->get_constant(stmt->tls_size)});
  }

  void create_offload_mesh_for(OffloadedStmt *stmt) override {
    auto *tls_prologue = create_mesh_xlogue(stmt->tls_prologue);

    llvm::Function *body;
    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
           llvm::Type::getInt8PtrTy(*llvm_context),
           tlctx->get_data_type<int>()});

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
#ifdef TI_LLVM_15
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
#else
        auto *loop_index_load = builder->CreateLoad(loop_index);
#endif
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
#ifdef TI_LLVM_15
        auto *loop_index_load =
            builder->CreateLoad(builder->getInt32Ty(), loop_index);
#else
        auto *loop_index_load = builder->CreateLoad(loop_index);
#endif
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

    create_call("gpu_parallel_mesh_for",
                {get_arg(0), tlctx->get_constant(stmt->num_cpu_threads),
                 tlctx->get_constant(stmt->mesh->num_patches),
                 tlctx->get_constant(stmt->block_dim), tls_prologue, body,
                 epilogue, tlctx->get_constant(stmt->tls_size)});
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
    stat.add("codegen_offloaded_tasks");
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
    if (stmt->bls_size > 0)
      create_bls_buffer(stmt);
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (prog->config.kernel_profiler && arch_is_cpu(prog->config.arch)) {
      call(
          builder.get(), "LLVMRuntime_profiler_start",
          {get_runtime(), builder->CreateGlobalStringPtr(offloaded_task_name)});
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
    if (prog->config.kernel_profiler && arch_is_cpu(prog->config.arch)) {
      llvm::IRBuilderBase::InsertPointGuard guard(*builder);
      builder->SetInsertPoint(final_block);
      call(builder.get(), "LLVMRuntime_profiler_stop", {get_runtime()});
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
};

}  // namespace

#ifdef TI_WITH_LLVM

static std::vector<uint8_t> generate_dxil_from_llvm(
    LLVMCompiledData &compiled_data,
    taichi::lang::Kernel *kernel){TI_NOT_IMPLEMENTED}

KernelCodeGenDX12::CompileResult KernelCodeGenDX12::compile() {
  TI_AUTO_PROF;
  auto *llvm_prog = get_llvm_program(prog);
  auto *tlctx = llvm_prog->get_llvm_context(kernel->arch);
  auto &config = prog->config;
  std::string kernel_key = get_hashed_offline_cache_key(&config, kernel);
  kernel->set_kernel_key_for_cache(kernel_key);

  if (!kernel->lowered()) {
    kernel->lower(/*to_executable=*/false);
  }

  auto block = dynamic_cast<Block *>(kernel->ir.get());
  TI_ASSERT(block);

  auto &offloads = block->statements;

  CompileResult Result;
  for (int i = 0; i < offloads.size(); i++) {
    auto offload =
        irpass::analysis::clone(offloads[i].get(), offloads[i]->get_kernel());
    irpass::re_id(offload.get());
    auto *offload_stmt = offload->as<OffloadedStmt>();
    auto new_data = compile_task(nullptr, offload_stmt);

    Result.task_dxil_source_codes.emplace_back(
        generate_dxil_from_llvm(new_data, kernel));
    aot::CompiledOffloadedTask task;
    // FIXME: build all fields for task.
    task.name = fmt::format("{}_{}_{}", kernel->get_name(),
                            offload_stmt->task_name(), i);
    task.type = offload_stmt->task_name();
    Result.tasks.emplace_back(task);
  }
  // FIXME: set correct num_snode_trees.
  Result.num_snode_trees = 1;
  return Result;
}

LLVMCompiledData KernelCodeGenDX12::compile_task(
    std::unique_ptr<llvm::Module> &&module,
    OffloadedStmt *stmt) {
  TaskCodeGenLLVMDX12 gen(kernel, stmt);
  return gen.run_compilation();
}
#endif  // TI_WITH_LLVM

FunctionType KernelCodeGenDX12::compile_to_function() {
  // FIXME: implement compile_to_function.
  return [](RuntimeContext &ctx) {};
}
TLANG_NAMESPACE_END
