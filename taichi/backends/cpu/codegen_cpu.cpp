#include "taichi/backends/cpu/codegen_cpu.h"

#include "taichi/codegen/codegen_llvm.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/util/statistics.h"

TLANG_NAMESPACE_BEGIN

class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  using IRVisitor::visit;

  CodeGenLLVMCPU(Kernel *kernel, IRNode *ir) : CodeGenLLVM(kernel, ir) {
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
          {llvm::PointerType::get(get_runtime_type("Context"), 0),
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
    create_call(
        "cpu_parallel_range_for",
        {get_arg(0), tlctx->get_constant(stmt->num_cpu_threads), begin, end,
         tlctx->get_constant(step), tlctx->get_constant(stmt->block_dim),
         tls_prologue, body, epilogue, tlctx->get_constant(stmt->tls_size)});
  }

  void visit(OffloadedStmt *stmt) override {
    stat.add("codegen_offloaded_tasks");
    TI_ASSERT(current_offload == nullptr);
    current_offload = stmt;
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
      call(builder.get(), "LLVMRuntime_profiler_stop", {get_runtime()});
    }
    finalize_offloaded_task_function();
    current_task->end();
    current_task = nullptr;
    current_offload = nullptr;
  }

  void visit(ExternalFuncCallStmt *stmt) override {
    std::vector<llvm::Type *> arg_types;
    std::vector<llvm::Value *> arg_values;

    for (auto s : stmt->arg_stmts) {
      TI_ASSERT(s->width() == 1);
      arg_types.push_back(tlctx->get_data_type(s->ret_type));
      arg_values.push_back(llvm_val[s]);
    }

    for (auto s : stmt->output_stmts) {
      TI_ASSERT(s->width() == 1);
      auto t = tlctx->get_data_type(s->ret_type);
      auto ptr = llvm::PointerType::get(t, 0);
      arg_types.push_back(ptr);
      arg_values.push_back(llvm_val[s]);
    }

    auto func_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(*llvm_context), arg_types, false);
    auto func_ptr_type = llvm::PointerType::get(func_type, 0);

    auto addr = tlctx->get_constant((std::size_t)stmt->func);
    auto func = builder->CreateIntToPtr(addr, func_ptr_type);
    builder->CreateCall(func, arg_values);
  }
};

FunctionType CodeGenCPU::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMCPU(kernel, ir).gen();
}

TLANG_NAMESPACE_END
