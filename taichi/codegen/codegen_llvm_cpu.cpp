#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include "taichi/tlang_util.h"
#include "taichi/program.h"
#include "taichi/ir.h"
#include "codegen_x86.h"
#include "codegen_llvm.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;
class CodeGenLLVMCPU : public CodeGenLLVM {
 public:
  using IRVisitor::visit;

  CodeGenLLVMCPU(Kernel *kernel) : CodeGenLLVM(kernel) {
  }

  void visit(OffloadedStmt *stmt) override {
    using Type = OffloadedStmt::TaskType;
    auto offloaded_task_name = init_offloaded_task_function(stmt);
    if (prog->config.enable_profiler) {
      call(
          builder.get(), "Runtime_profiler_start",
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
      call(builder.get(), "Runtime_profiler_stop", {get_runtime()});
    }
    finalize_offloaded_task_function();
    current_task->end();
    current_task = nullptr;
  }
};

FunctionType CPUCodeGen::codegen_llvm() {
  TI_PROFILER("cpu codegen");
  return CodeGenLLVMCPU(kernel).gen();
}

TLANG_NAMESPACE_END
