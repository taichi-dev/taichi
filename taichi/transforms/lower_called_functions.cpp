#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/statements.h"
#include "taichi/program/function.h"
#include "taichi/program/compile_config.h"

namespace taichi::lang {

class LowerCalledFunctions : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit LowerCalledFunctions(const CompileConfig &compile_config)
      : compile_config_(compile_config) {
  }

  void visit(FuncCallStmt *stmt) override {
    auto *func = stmt->func;
    if (!func->lowered()) {
      func->set_lowered(true);
      irpass::compile_function(func->ir.get(), compile_config_, func,
                               /*autodiff_mode=*/AutodiffMode::kNone,
                               /*verbose=*/compile_config_.print_ir,
                               /*start_from_ast=*/func->ir_start_from_ast());
      func->ir->accept(this);
    }
  }

  static void run(const CompileConfig &compile_config, IRNode *root) {
    LowerCalledFunctions lcf{compile_config};
    root->accept(&lcf);
  }

 private:
  const CompileConfig &compile_config_;
};

namespace irpass {

void lower_called_functions(const CompileConfig &compile_config, IRNode *root) {
  TI_AUTO_PROF;
  LowerCalledFunctions::run(compile_config, root);
}

}  // namespace irpass

}  // namespace taichi::lang
