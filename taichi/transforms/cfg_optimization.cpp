#include "taichi/ir/ir.h"
#include "taichi/ir/cfg.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/analysis.h"

TLANG_NAMESPACE_BEGIN

namespace irpass {
void cfg_optimization(IRNode *root) {
  auto cfg = analysis::build_cfg(root);
  while (cfg->unreachable_code_elimination());
}
} // namespace irpass

TLANG_NAMESPACE_END
