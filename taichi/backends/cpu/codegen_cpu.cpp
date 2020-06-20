#include "codegen_cpu.h"

#include "taichi/codegen/codegen_llvm.h"
#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/util/statistics.h"

TLANG_NAMESPACE_BEGIN

class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  using IRVisitor::visit;

  CodeGenLLVMCPU(Kernel *kernel, IRNode *ir)
      : CodeGenLLVM(kernel, ir){TI_AUTO_PROF}

        llvm::Value
        *
        get_tls_base_ptr() {
    return get_arg(1);
  }

  void visit(ThreadLocalPtrStmt *stmt) override {
    auto base = get_tls_base_ptr();
    TI_ASSERT(stmt->width() == 1);
    TI_P(type_name(base->getType()));
    auto ptr = builder->CreateGEP(base, tlctx->get_constant(stmt->offset));
    TI_P(type_name(ptr->getType()));
    auto ptr_type = llvm::PointerType::get(
        tlctx->get_data_type(stmt->ret_type.data_type), 0);
    TI_P(type_name(ptr_type));
    llvm_val[stmt] = builder->CreatePointerCast(ptr, ptr_type);
  }

  void create_offload_range_for(OffloadedStmt *stmt) override {
    int step = 1;
    if (stmt->reversed) {
      step = -1;
    }

    auto tls_ptr_type = llvm::Type::getInt8PtrTy(*llvm_context);

    std::vector<llvm::Type *> xlogue_arguments{
        llvm::PointerType::get(get_runtime_type("Context"), 0), tls_ptr_type};

    auto xlogue_type = llvm::FunctionType::get(
        llvm::Type::getVoidTy(*llvm_context), xlogue_arguments, false);
    auto xlogue_ptr_type = llvm::PointerType::get(xlogue_type, 0);

    llvm::Value *prologue = nullptr;
    if (stmt->prologue) {
      auto guard = get_function_creation_guard(xlogue_arguments);

      stmt->prologue->accept(this);

      prologue = guard.body;
    } else {
      prologue = llvm::ConstantPointerNull::get(xlogue_ptr_type);
    }

    llvm::Function *body;

    {
      auto guard = get_function_creation_guard(
          {llvm::PointerType::get(get_runtime_type("Context"), 0),
           llvm::Type::getInt8PtrTy(*llvm_context),
           tlctx->get_data_type<int>()});

      auto loop_var = create_entry_block_alloca(DataType::i32);
      loop_vars_llvm[stmt].push_back(loop_var);
      builder->CreateStore(get_arg(2), loop_var);
      stmt->body->accept(this);

      body = guard.body;
    }

    llvm::Value *epilogue = nullptr;
    if (stmt->epilogue) {
      auto guard = get_function_creation_guard(xlogue_arguments);

      stmt->epilogue->accept(this);

      epilogue = guard.body;
    } else {
      epilogue = llvm::ConstantPointerNull::get(xlogue_ptr_type);
    }

    auto [begin, end] = get_range_for_bounds(stmt);
    create_call(
        "cpu_parallel_range_for",
        {get_arg(0), tlctx->get_constant(stmt->num_cpu_threads), begin, end,
         tlctx->get_constant(step), tlctx->get_constant(stmt->block_dim),
         prologue, body, epilogue});
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
      stmt->block_dim =
          std::min(stmt->snode->parent->max_num_elements(), stmt->block_dim);
      create_offload_struct_for(stmt);
    } else if (stmt->task_type == Type::clear_list) {
      emit_clear_list(stmt);
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
};

FunctionType CodeGenCPU::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMCPU(kernel, ir).gen();
}

TLANG_NAMESPACE_END
