#include "taichi/ir/analysis.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/program/async_utils.h"
#include "taichi/program/ir_bank.h"

TLANG_NAMESPACE_BEGIN

namespace irpass::analysis {
void get_meta_input_value_states(IRNode *root,
                                 TaskMeta *meta,
                                 IRBank *ir_bank) {
  auto cfg = analysis::build_cfg(root);
  auto snodes = cfg->gather_loaded_snodes();
  for (auto &snode : snodes) {
    meta->input_states.insert(
        ir_bank->get_async_state(snode, AsyncState::Type::value));
  }
}
}  // namespace irpass::analysis
TLANG_NAMESPACE_END
