#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
void cfg_optimization(IRNode *root, bool after_lower_access) {
  TI_AUTO_PROF;
  auto cfg = analysis::build_cfg(root);
  while (true) {
    bool modified = false;
    cfg->simplify_graph();
    if (cfg->store_to_load_forwarding(after_lower_access))
      modified = true;
    std::cout << "before:" << std::endl;
    print(root);
    if (cfg->dead_store_elimination(after_lower_access))
      modified = true;
    std::cout << "after:" << std::endl;
    print(root);
    if (!modified)
      break;
  }
}
}  // namespace irpass

TLANG_NAMESPACE_END
