#include "taichi/program/async_utils.h"

#include "taichi/ir/analysis.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

void TaskMeta::remove_duplication() {
  std::sort(input_states.begin(), input_states.end());
  input_states.resize(
      std::size_t(std::unique(input_states.begin(), input_states.end()) -
                  input_states.begin()));
  std::sort(output_states.begin(), output_states.end());
  output_states.resize(
      std::size_t(std::unique(output_states.begin(), output_states.end()) -
                  output_states.begin()));
}

std::unique_ptr<IRNode> IRHandle::clone() const {
  // TODO: remove get_kernel() here
  return irpass::analysis::clone(const_cast<IRNode *>(ir_), ir_->get_kernel());
}

TaskLaunchRecord::TaskLaunchRecord() : kernel(nullptr), ir_handle(nullptr, 0) {
}

TaskLaunchRecord::TaskLaunchRecord(Context context,
                                   Kernel *kernel,
                                   IRHandle ir_handle)
    : context(context), kernel(kernel), ir_handle(ir_handle) {
  TI_ASSERT(ir_handle.ir()->get_kernel() != nullptr);
}

OffloadedStmt *TaskLaunchRecord::stmt() const {
  TI_ASSERT(ir_handle.ir());
  return const_cast<IRNode *>(ir_handle.ir())->as<OffloadedStmt>();
}

TLANG_NAMESPACE_END
