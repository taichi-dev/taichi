#include "taichi/ir/ir.h"
#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {

namespace {

std::function<void(const std::string &)> make_pass_printer(IRNode *ir) {
  return [ir](const std::string &pass) {
    TI_INFO("[debug] {}:", pass);
    std::cout << std::flush;
    irpass::re_id(ir);
    irpass::print(ir);
    std::cout << std::flush;
  };
}

}  // namespace

namespace irpass {
bool cfg_optimization(
    IRNode *root,
    bool after_lower_access,
    bool autodiff_enabled,
    bool real_matrix_enabled,
    const std::optional<ControlFlowGraph::LiveVarAnalysisConfig>
        &lva_config_opt) {
  TI_AUTO_PROF;
  auto print = make_pass_printer(root);
  auto cfg = analysis::build_cfg(root);
  bool result_modified = false;
  if (!real_matrix_enabled) {
    cfg->simplify_graph();

    if (cfg->store_to_load_forwarding(after_lower_access, autodiff_enabled)) {
      result_modified = true;
      print("store_to_load_forwarding");
    }

    if (cfg->dead_store_elimination(after_lower_access, lva_config_opt)) {
      result_modified = true;
      print("dead_store_elimination");
    }
  }
  // TODO: implement cfg->dead_instruction_elimination()
  if (die(root))  // remove unused allocas
    print("dead_store_elimination");
  return result_modified;
}
}  // namespace irpass

}  // namespace taichi::lang
