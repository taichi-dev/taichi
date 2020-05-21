#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"
#include <deque>
#include <set>
#include <cmath>
#include <thread>

#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN

class ConstantFold : public BasicStmtVisitor {
 public:
  using BasicStmtVisitor::visit;

  ConstantFold() : BasicStmtVisitor() {
  }

  static Kernel *get_jit_evaluator_kernel(JITEvaluatorId const &id) {
    auto &cache = get_current_program().jit_evaluator_cache;
    {
      // Discussion:
      // https://github.com/taichi-dev/taichi/pull/954#discussion_r423442606
      std::lock_guard<std::mutex> _(
          get_current_program().jit_evaluator_cache_mut);
      auto it = cache.find(id);
      if (it != cache.end())  // cached?
        return it->second.get();
    }

    auto kernel_name = fmt::format("jit_evaluator_{}", cache.size());
    auto func = [&]() {
      auto lhstmt = Stmt::make<ArgLoadStmt>(0, false);
      auto rhstmt = Stmt::make<ArgLoadStmt>(1, false);
      pStmt oper;
      if (id.is_binary) {
        oper = Stmt::make<BinaryOpStmt>(id.binary_op(), lhstmt.get(),
                                        rhstmt.get());
      } else {
        oper = Stmt::make<UnaryOpStmt>(id.unary_op(), lhstmt.get());
        if (unary_op_is_cast(id.unary_op())) {
          oper->cast<UnaryOpStmt>()->cast_type = id.rhs;
        }
      }
      auto ret = Stmt::make<KernelReturnStmt>(oper.get());
      current_ast_builder().insert(std::move(lhstmt));
      if (id.is_binary)
        current_ast_builder().insert(std::move(rhstmt));
      current_ast_builder().insert(std::move(oper));
      current_ast_builder().insert(std::move(ret));
    };
    auto ker =
        std::make_unique<Kernel>(get_current_program(), func, kernel_name);
    ker->insert_ret(id.ret);
    ker->insert_arg(id.lhs, false);
    if (id.is_binary)
      ker->insert_arg(id.rhs, false);
    ker->is_evaluator = true;
    auto *ker_ptr = ker.get();
    TI_TRACE("Saving JIT evaluator cache entry id={}",
             std::hash<JITEvaluatorId>{}(id));
    {
      std::lock_guard<std::mutex> _(
          get_current_program().jit_evaluator_cache_mut);
      cache[id] = std::move(ker);
    }
    return ker_ptr;
  }

  static bool is_good_type(DataType dt) {
    // ConstStmt of `bad` types like `i8` is not supported by LLVM.
    // Discussion:
    // https://github.com/taichi-dev/taichi/pull/839#issuecomment-625902727
    switch (dt) {
      case DataType::i32:
      case DataType::f32:
      case DataType::i64:
      case DataType::f64:
        return true;
      default:
        return false;
    }
  }

  class ContextArgSaveGuard {
    Context &ctx;
    uint64 old_args[taichi_max_num_args];

   public:
    explicit ContextArgSaveGuard(Context &ctx_) : ctx(ctx_) {
      std::memcpy(old_args, ctx.args, sizeof(old_args));
    }

    ~ContextArgSaveGuard() {
      std::memcpy(ctx.args, old_args, sizeof(old_args));
    }
  };

  static bool jit_evaluate_binary_op(TypedConstant &ret,
                                     BinaryOpStmt *stmt,
                                     const TypedConstant &lhs,
                                     const TypedConstant &rhs) {
    if (!is_good_type(ret.dt))
      return false;
    JITEvaluatorId id{std::this_thread::get_id(),
                      (int)stmt->op_type,
                      ret.dt,
                      lhs.dt,
                      rhs.dt,
                      true};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().get_context();
    ContextArgSaveGuard _(
        ctx);  // save input args, prevent override current kernel
    ctx.set_arg<int64_t>(0, lhs.val_i64);
    ctx.set_arg<int64_t>(1, rhs.val_i64);
    (*ker)();
    ret.val_i64 = get_current_program().fetch_result<int64_t>(0);
    return true;
  }

  static bool jit_evaluate_unary_op(TypedConstant &ret,
                                    UnaryOpStmt *stmt,
                                    const TypedConstant &operand) {
    if (!is_good_type(ret.dt))
      return false;
    JITEvaluatorId id{std::this_thread::get_id(),
                      (int)stmt->op_type,
                      ret.dt,
                      operand.dt,
                      stmt->cast_type,
                      false};
    auto *ker = get_jit_evaluator_kernel(id);
    auto &ctx = get_current_program().get_context();
    ContextArgSaveGuard _(
        ctx);  // save input args, prevent override current kernel
    ctx.set_arg<int64_t>(0, operand.val_i64);
    (*ker)();
    ret.val_i64 = get_current_program().fetch_result<int64_t>(0);
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
    if (jit_evaluate_binary_op(new_constant, stmt, lhs->val[0], rhs->val[0])) {
      auto evaluated =
          Stmt::make<ConstStmt>(LaneAttribute<TypedConstant>(new_constant));
      stmt->replace_with(evaluated.get());
      stmt->parent->insert_before(stmt, VecStatement(std::move(evaluated)));
      stmt->parent->erase(stmt);
      throw IRModified();
    }
  }

  void visit(UnaryOpStmt *stmt) override {
    auto operand = stmt->operand->cast<ConstStmt>();
    if (!operand)
      return;
    if (stmt->width() != 1)
      return;
    auto dst_type = stmt->ret_type.data_type;
    TypedConstant new_constant(dst_type);
    if (jit_evaluate_unary_op(new_constant, stmt, operand->val[0])) {
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
  // @archibate found that `debug=True` will cause JIT kernels
  // failed to evaluate correctly (always return 0), so we simply
  // disable constant_fold when config.debug is turned on.
  // Discussion:
  // https://github.com/taichi-dev/taichi/pull/839#issuecomment-626107010
  if (get_current_program().config.debug) {
    TI_TRACE("config.debug enabled, ignoring constant fold");
    return;
  }
  if (!advanced_optimization)
    return;
  return ConstantFold::run(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
