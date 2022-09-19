#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/system/profiler.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
bool cfg_optimization(
    IRNode *root,
    bool after_lower_access,
    bool autodiff_enabled,
    bool real_matrix_enabled,
    const std::optional<ControlFlowGraph::LiveVarAnalysisConfig>
        &lva_config_opt) {
  TI_AUTO_PROF;
  auto cfg = analysis::build_cfg(root);
  bool result_modified = false;
  while (true && !real_matrix_enabled) {
    bool modified = false;
    cfg->simplify_graph();
    if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled))
      modified = true;
    if (cfg->dead_store_elimination(after_lower_access, lva_config_opt))
      modified = true;
    if (modified)
      result_modified = true;
    else
      break;
  }
  // TODO: implement cfg->dead_instruction_elimination()
  die(root);  // remove unused allocas
  return result_modified;
}
}  // namespace irpass

TLANG_NAMESPACE_END
