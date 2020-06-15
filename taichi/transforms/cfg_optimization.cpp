#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
void cfg_optimization(IRNode *root) {
  auto cfg = analysis::build_cfg(root);
  while (true) {
    bool modified = false;
    cfg->simplify_graph();
    if (cfg->unreachable_code_elimination())
      modified = true;
    if (cfg->store_to_load_forwarding())
      modified = true;
    if (!modified)
      break;
  }
}
}  // namespace irpass

TLANG_NAMESPACE_END
