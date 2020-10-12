#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/program/async_utils.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {
void get_meta_input_value_states(IRNode *root, TaskMeta *meta) {
  auto cfg = analysis::build_cfg(root);
  auto snodes = cfg->gather_loaded_snodes();
  for (auto &snode : snodes) {
    meta->input_states.emplace(snode, AsyncState::Type::value);
  }
}
}  // namespace irpass::analysis
TLANG_NAMESPACE_END
