#include "codegen_cpu.h"

#include "taichi/codegen/codegen_llvm.h"
#include "taichi/common/util.h"
#include "taichi/util/io.h"
#include "taichi/lang_util.h"
#include "taichi/program/program.h"
#include "taichi/ir/ir.h"
#include "taichi/util/statistics.h"

TLANG_NAMESPACE_BEGIN

class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  using IRVisitor::visit;

  CodeGenLLVMCPU(Kernel *kernel, IRNode *ir) : CodeGenLLVM(kernel, ir) {
    TI_AUTO_PROF
  }

  void visit(OffloadedStmt *stmt) override {
    stat.add("codegen_offloaded_tasks");
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (prog->config.enable_profiler) {
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
    if (prog->config.enable_profiler) {
      call(builder.get(), "LLVMRuntime_profiler_stop", {get_runtime()});
    }
    finalize_offloaded_task_function();
    current_task->end();
    current_task = nullptr;
  }
};

FunctionType CodeGenCPU::codegen() {
  TI_AUTO_PROF
  return CodeGenLLVMCPU(kernel, ir).gen();
}

TLANG_NAMESPACE_END
