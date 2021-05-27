#include "taichi/ir/scratch_pad.h"

#include "taichi/ir/statements.h"

TLANG_NAMESPACE_BEGIN

std::string ScratchPad::global_to_linearized_local(
    const std::vector<Stmt *> &loop_vars,
    const std::vector<Stmt *> &indices) {
  std::string ret = "";
  TI_ASSERT((int)indices.size() == dim);
  int step_size = pad_size_linear();
  for (int i = 0; i < (int)indices.size(); i++) {
    TI_ASSERT(step_size % pad_size[i] == 0);
    step_size /= pad_size[i];
    ret += fmt::format(" + ({} - {}_base - {}) * {}", indices[i]->raw_name(),
                       loop_vars[i]->raw_name(), bounds[i].low, step_size);
  }
  return ret;
}

TLANG_NAMESPACE_END
