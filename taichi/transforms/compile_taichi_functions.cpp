#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/ir/statements.h"
#include "taichi/program/function.h"
#include "taichi/program/compile_config.h"

namespace taichi::lang {

class CompileTaichiFunctions : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  explicit CompileTaichiFunctions(const CompileConfig &compile_config)
      : compile_config_(compile_config) {
  }

  void visit(FuncCallStmt *stmt) override {
    using IRType = Function::IRType;
    auto *func = stmt->func;
    const auto ir_type = func->ir_type();
    if (ir_type != IRType::OptimizedIR) {
      TI_ASSERT(ir_type == IRType::AST || ir_type == IRType::InitialIR);
      func->set_ir_type(IRType::OptimizedIR);
      irpass::compile_function(func->ir.get(), compile_config_, func,
                               /*autodiff_mode=*/AutodiffMode::kNone,
                               /*verbose=*/compile_config_.print_ir,
                               /*start_from_ast=*/ir_type == IRType::AST);
      func->ir->accept(this);
    }
  }

  static void run(IRNode *ir, const CompileConfig &compile_config) {
    CompileTaichiFunctions ctf{compile_config};
    ir->accept(&ctf);
  }

 private:
  const CompileConfig &compile_config_;
};

namespace irpass {

void compile_taichi_functions(IRNode *ir, const CompileConfig &compile_config) {
  TI_AUTO_PROF;
  CompileTaichiFunctions::run(ir, compile_config);
}

}  // namespace irpass

}  // namespace taichi::lang
