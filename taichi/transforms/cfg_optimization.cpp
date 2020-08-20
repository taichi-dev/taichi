#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
bool cfg_optimization(IRNode *root, bool after_lower_access) {
  TI_AUTO_PROF;
  auto cfg = analysis::build_cfg(root);
  bool result_modified = false;
  while (true) {
    bool modified = false;
    cfg->simplify_graph();
    std::cout << "before store-to-load" << std::endl;
    print(root);
    if (cfg->store_to_load_forwarding(after_lower_access))
      modified = true;
    std::cout << "after store-to-load" << std::endl;
    print(root);
    if (cfg->dead_store_elimination(after_lower_access))
      modified = true;
    std::cout << "after store" << std::endl;
    print(root);
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
