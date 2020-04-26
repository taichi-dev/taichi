#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/ir/snode.h"
#include <deque>
#include <set>
#include <cmath>

TLANG_NAMESPACE_BEGIN

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ConstantFold() : BasicStmtVisitor() {
  }

  void visit(UnaryOpStmt *stmt) override {
    return; // TODO TODO
#if 0       // TODO TODO
    if (stmt->width() == 1 && stmt->op_type == UnaryOpType::cast &&
        stmt->cast_by_value && stmt->operand->is<ConstStmt>()) {
      auto input = stmt->operand->as<ConstStmt>()->val[0];
      auto src_type = stmt->operand->ret_type.data_type;
      auto dst_type = stmt->ret_type.data_type;
      TypedConstant new_constant(dst_type);
      bool success = false;
      if (src_type == DataType::f32) {
        auto v = input.val_float32();
        if (dst_type == DataType::i32) {
          new_constant.val_i32 = int32(v);
          success = true;
        }
      } else if (src_type == DataType::i32) {
        auto v = input.val_int32();
        if (dst_type == DataType::f32) {
          new_constant.val_f32 = float32(v);
          success = true;
        }
      }

      if (success) {
        auto evaluated =
            Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
        stmt->replace_with(evaluated.get());
        stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
        stmt->parent->erase(stmt);
        throw IRModified();
      }
    }
#endif
  }

  static Kernel *get_binary_op_jit_eval_kernel(BinaryOpType op)
  {
    static std::map<BinaryOpType, std::unique_ptr<Kernel>> cache; // X: race?
    auto it = cache.find(op);
    if (it != cache.end()) // cached?
      return it->second.get();
    static int jic = 0; // X: race?
    auto kernel_name = fmt::format("jit_constexpr_{}", jic++);
    auto func = [&] () {
      auto lhstmt = Stmt::make<ArgLoadStmt>(1, false);
      auto rhstmt = Stmt::make<ArgLoadStmt>(2, false);
      auto oper = Stmt::make<BinaryOpStmt>(op, lhstmt.get(), rhstmt.get());
      auto ret = Stmt::make<ArgStoreStmt>(0, oper.get());
      current_ast_builder().insert(std::move(lhstmt));
      current_ast_builder().insert(std::move(rhstmt));
      current_ast_builder().insert(std::move(oper));
      current_ast_builder().insert(std::move(ret));
    };
    auto ker = std::make_unique<Kernel>(get_current_program(), func, kernel_name);
    ker->insert_arg(DataType::i32, false); // ret
    ker->insert_arg(DataType::i32, false); // lhstmt
    ker->insert_arg(DataType::i32, false); // rhstmt
    ker->mark_arg_return_value(0, true);
    auto *ker_p = ker.get();
    cache[op] = std::move(ker);
    return ker_p;
  }

  static bool jit_from_binary_op(TypedConstant &tc, BinaryOpType op,
      const TypedConstant &lhs, const TypedConstant &rhs)
  {
    if (lhs.dt != DataType::i32 || rhs.dt != DataType::i32 || tc.dt != DataType::i32)
      return false;
    auto ker = get_binary_op_jit_eval_kernel(BinaryOpType::cmp_ge);
    get_current_program().context.set_arg<int>(1, lhs.val_i32);
    get_current_program().context.set_arg<int>(2, rhs.val_i32);
    TI_INFO("!!!IN");
    irpass::print(ker->ir);
    (*ker)();
    auto ret = get_current_program().context.get_arg<int>(0);
    TI_INFO("!!!OUT {}", ret);
    tc.val_i32 = ret;
    return true;
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs = stmt->lhs->cast<ConstStmt>();
    auto rhs = stmt->rhs->cast<ConstStmt>();
    if (!lhs || !rhs)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type.data_type;
    TypedConstant new_constant(dst_type);
    if (jit_from_binary_op(new_constant, stmt->op_type, lhs->val[0], rhs->val[0])) {
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  static void run(IRNode *node) {
    ConstantFold folder;
    while (true) {
      bool modified = false;
      try {
        node->accept(&folder);
      } catch (IRModified) {
        modified = true;
      }
      if (!modified)
        break;
    }
  }
};

namespace irpass {

void constant_fold(IRNode *root) {
  return ConstantFold::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
