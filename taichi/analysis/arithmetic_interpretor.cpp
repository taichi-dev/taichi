#include "taichi/analysis/arithmetic_interpretor.h"

#include <algorithm>
#include <type_traits>
#include <vector>

#include "taichi/ir/type_utils.h"
#include "taichi/ir/visitors.h"

namespace taichi {
namespace lang {
namespace {

using CodeRegion = ArithmeticInterpretor::CodeRegion;
using EvalContext = ArithmeticInterpretor::EvalContext;

std::vector<Stmt *> get_raw_statements(const Block *block) {
  const auto &stmts = block->statements;
  std::vector<Stmt *> res(stmts.size());
  std::transform(stmts.begin(), stmts.end(), res.begin(),
                 [](const std::unique_ptr<Stmt> &s) { return s.get(); });
  return res;
}

class EvalVisitor : public IRVisitor {
 public:
  explicit EvalVisitor() {
    allow_undefined_visitor = true;
    invoke_default_visitor = true;
  }

  std::optional<TypedConstant> run(const CodeRegion &region,
                                   const EvalContext &init_ctx) {
    context_ = init_ctx;
    failed_ = false;

    auto stmts = get_raw_statements(region.block);
    if (stmts.empty()) {
      return std::nullopt;
    }
    auto *begin_stmt = (region.begin == nullptr) ? stmts.front() : region.begin;
    auto *end_stmt = (region.end == nullptr) ? stmts.back() : region.end;

    auto cur_iter = std::find(stmts.begin(), stmts.end(), begin_stmt);
    auto end_iter = std::find(stmts.begin(), stmts.end(), end_stmt);
    if ((cur_iter == stmts.end()) || (end_iter == stmts.end())) {
      return std::nullopt;
    }
    Stmt *cur_stmt = nullptr;
    while (cur_iter != end_iter) {
      cur_stmt = *cur_iter;
      cur_stmt->accept(this);
      if (failed_) {
        return std::nullopt;
      }
      ++cur_iter;
    }
    return context_.maybe_get(cur_stmt);
  }

  void visit(ConstStmt *stmt) override {
    context_.insert(stmt, stmt->val);
  }

  void visit(BinaryOpStmt *stmt) override {
    auto lhs_opt = context_.maybe_get(stmt->lhs);
    auto rhs_opt = context_.maybe_get(stmt->rhs);
    if (!lhs_opt || !rhs_opt) {
      failed_ = true;
      return;
    }
    auto lhs = lhs_opt.value();
    auto rhs = rhs_opt.value();
    if (lhs.dt != rhs.dt) {
      failed_ = true;
      return;
    }

    const auto op = stmt->op_type;
    const auto dt = lhs.dt;
    // TODO: Consider using macros to avoid duplication
    if (is_real(dt)) {
      // Put floating point numbers first because is_signed/unsigned asserts
      // that the data type being integral.
      auto res_opt = eval_bin_op(lhs.val_float(), rhs.val_float(), op);
      insert_or_failed(stmt, dt, res_opt);
    } else if (is_signed(dt)) {
      auto res_opt = eval_bin_op(lhs.val_int(), rhs.val_int(), op);
      insert_or_failed(stmt, dt, res_opt);
    } else if (is_unsigned(dt)) {
      auto res_opt = eval_bin_op(lhs.val_uint(), rhs.val_uint(), op);
      insert_or_failed(stmt, dt, res_opt);
    } else {
      TI_NOT_IMPLEMENTED;
      failed_ = true;
    }
  }

  void visit(BitExtractStmt *stmt) override {
    auto val_opt = context_.maybe_get(stmt->input);
    if (!val_opt) {
      failed_ = true;
      return;
    }
    const uint64_t mask = (1ULL << (stmt->bit_end - stmt->bit_begin)) - 1;
    auto val = val_opt.value().val_int();
    val = (val >> stmt->bit_begin) & mask;
    insert_to_ctx(stmt, stmt->ret_type, val);
  }

  void visit(LinearizeStmt *stmt) override {
    int64_t val = 0;
    for (int i = 0; i < (int)stmt->inputs.size(); ++i) {
      auto idx_opt = context_.maybe_get(stmt->inputs[i]);
      if (!idx_opt) {
        failed_ = true;
        return;
      }
      val = (val * stmt->strides[i]) + idx_opt.value().val_int();
    }
    insert_to_ctx(stmt, stmt->ret_type, val);
  }

  void visit(Stmt *stmt) override {
    if (context_.should_ignore(stmt)) {
      return;
    }
    failed_ = (context_.maybe_get(stmt) == std::nullopt);
  }

 private:
  template <typename T>
  static std::optional<T> eval_bin_op(T lhs, T rhs, BinaryOpType op) {
    if (op == BinaryOpType::add) {
      return lhs + rhs;
    }
    if (op == BinaryOpType::sub) {
      return lhs - rhs;
    }
    if (op == BinaryOpType::mul) {
      return lhs * rhs;
    }
    if (op == BinaryOpType::div) {
      return lhs / rhs;
    }
    if constexpr (std::is_integral_v<T>) {
      if (op == BinaryOpType::mod) {
        return lhs % rhs;
      }
    }
    return std::nullopt;
  }

  template <typename T>
  void insert_or_failed(const Stmt *stmt,
                        DataType dt,
                        std::optional<T> val_opt) {
    if (!val_opt) {
      failed_ = true;
      return;
    }
    context_.insert(stmt, TypedConstant(dt, val_opt.value()));
  }

  template <typename T>
  void insert_to_ctx(const Stmt *stmt, DataType dt, const T &val) {
    context_.insert(stmt, TypedConstant(dt, val));
  }

  EvalContext context_;
  bool failed_{false};
};

}  // namespace

std::optional<TypedConstant> ArithmeticInterpretor::evaluate(
    const CodeRegion &region,
    const EvalContext &init_ctx) const {
  EvalVisitor ev;
  return ev.run(region, init_ctx);
}

}  // namespace lang
}  // namespace taichi
