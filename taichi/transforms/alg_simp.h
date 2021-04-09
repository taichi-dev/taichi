#pragma once

#include "taichi/ir/pass.h"

namespace taichi {
namespace lang {

class AlgSimpPass : public Pass {
 public:
  static const PassID id;
  Status run(const PassContext &ctx,
             IRNode *module,
             AnalysisManager *amgr) override;
};
const PassID AlgSimpPass::id = "AlgSimpPass";

}  // namespace lang
}  // namespace taichi
