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

  struct BinaryEvaluatorId
  {
    BinaryOpType op;
    DataType ret, lhs, rhs;

    explicit operator int() const // make STL map happy
    {
      return (int)op | (int) ret << 8 | (int) lhs << 16 | (int) rhs << 24;
    }
  };

  static Kernel *get_binary_op_jit_eval_kernel(BinaryEvaluatorId const &id)
  {
    auto &cache = get_current_program().jit_evaluator_cache;
#if 1
    int iid = int(id);
    auto it = cache.find(iid);
    if (it != cache.end()) // cached?
      return it->second.get();
#endif
    static int jic = 0; // X: race?
    auto kernel_name = fmt::format("jit_evaluator_{}", jic++);
    auto func = [&] () {
      auto lhstmt = Stmt::make<ArgLoadStmt>(1, false);
      auto rhstmt = Stmt::make<ArgLoadStmt>(2, false);
      auto oper = Stmt::make<BinaryOpStmt>(id.op, lhstmt.get(), rhstmt.get());
      auto ret = Stmt::make<ArgStoreStmt>(0, oper.get());
      current_ast_builder().insert(std::move(lhstmt));
      current_ast_builder().insert(std::move(rhstmt));
      current_ast_builder().insert(std::move(oper));
      current_ast_builder().insert(std::move(ret));
    };
    auto ker = std::make_unique<Kernel>(get_current_program(), func, kernel_name);
    ker->insert_arg(id.ret, false);
    ker->insert_arg(id.lhs, false);
    ker->insert_arg(id.rhs, false);
    ker->mark_arg_return_value(0, true);
    auto *ker_ptr = ker.get();
#if 1
    cache[iid] = std::move(ker);
#endif
    return ker_ptr;
  }

  static bool jit_from_binary_op(TypedConstant &ret, BinaryOpType op,
      const TypedConstant &lhs, const TypedConstant &rhs)
  {
    BinaryEvaluatorId id{op, ret.dt, lhs.dt, rhs.dt};
    auto *ker = get_binary_op_jit_eval_kernel(id);
    auto &ctx = get_current_program().context;
#define PER_TYPE(x) ctx.set_arg(1, lhs.val_##x);
    PER_TYPE(i32) PER_TYPE(i64)
    PER_TYPE(f32) PER_TYPE(f64)
#undef PER_TYPE
#define PER_TYPE(x) ctx.set_arg(2, rhs.val_##x);
    PER_TYPE(i32) PER_TYPE(i64)
    PER_TYPE(f32) PER_TYPE(f64)
#undef PER_TYPE
    irpass::print(ker->ir);
    (*ker)();
#define PER_TYPE(x, T) ret.val_##x = ctx.get_arg<T>(0);
    PER_TYPE(i32, int) PER_TYPE(i64, int64_t)
    PER_TYPE(f32, float) PER_TYPE(f64, double)
#undef PER_TYPE
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
