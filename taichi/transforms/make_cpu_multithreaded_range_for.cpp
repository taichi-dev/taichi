#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/utils.h"

namespace taichi::lang {

namespace {

using TaskType = OffloadedStmt::TaskType;

void make_multithreaded_range_for(OffloadedStmt *offloaded,
                                  const CompileConfig &config) {
  TI_ASSERT(offloaded->task_type == TaskType::range_for);

  auto offloaded_body = std::make_unique<Block>();
  offloaded_body->insert(
      Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 1)));
  auto one = offloaded_body->back();
  auto minimal_block_range = offloaded_body->insert(
      Stmt::make_typed<ConstStmt>(TypedConstant(PrimitiveType::i32, 512)));
  offloaded_body->insert(Stmt::make_typed<ConstStmt>(
      TypedConstant(PrimitiveType::i32, config.cpu_max_num_threads)));
  auto num_threads = offloaded_body->back();
  offloaded_body->insert(Stmt::make_typed<LoopIndexStmt>(offloaded, 0));
  auto thread_index = offloaded_body->back();

  // Retrieve range-for bounds.
  Stmt *begin_stmt;
  Stmt *end_stmt;
  if (offloaded->const_begin) {
    begin_stmt = offloaded_body->insert(Stmt::make_typed<ConstStmt>(
        TypedConstant(PrimitiveType::i32, offloaded->begin_value)));
  } else {
    begin_stmt = offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(
        offloaded->begin_offset, PrimitiveType::i32));
    begin_stmt = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(offloaded_body->back()));
  }
  if (offloaded->const_end) {
    end_stmt = offloaded_body->insert(Stmt::make_typed<ConstStmt>(
        TypedConstant(PrimitiveType::i32, offloaded->end_value)));
  } else {
    end_stmt =  offloaded_body->insert(Stmt::make<GlobalTemporaryStmt>(
        offloaded->end_offset, PrimitiveType::i32));
    end_stmt = offloaded_body->insert(Stmt::make<GlobalLoadStmt>(offloaded_body->back()));
  }

  // Inner serial block range is
  // $[max(((end - begin) + (num_threads - 1)) / num_threads, 1)]
  offloaded_body->insert(
      Stmt::make_typed<BinaryOpStmt>(BinaryOpType::sub, end_stmt, begin_stmt));
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::add, offloaded_body->back(), num_threads));
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::sub, offloaded_body->back(), one));
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::floordiv, offloaded_body->back(), num_threads));
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::max, offloaded_body->back(), minimal_block_range));

  // Inner loop begins at $[begin + block_range * thread_id]
  auto block_range = offloaded_body->back();
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::mul, block_range, thread_index));
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::add, begin_stmt, offloaded_body->back()));
  auto block_begin = offloaded_body->back();

  // Inner loop ends at $[min(block_begin + block_range), end))]
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::add, block_begin, block_range));
  offloaded_body->insert(Stmt::make_typed<BinaryOpStmt>(
      BinaryOpType::min, end_stmt, offloaded_body->back()));
  auto block_end = offloaded_body->back();

  offloaded_body->insert(Stmt::make_typed<RangeForStmt>(
      block_begin, block_end, std::move(offloaded->body), false, 1, 1,
      /*strictly_serialized*/ true, offloaded->range_hint));
  auto inner_loop = offloaded_body->back();

  irpass::replace_all_usages_with(inner_loop, offloaded, inner_loop);

  // Update the offloaded stmt.
  offloaded->const_begin = true;
  offloaded->const_end = true;
  offloaded->begin_value = 0;
  offloaded->end_value = config.cpu_max_num_threads;
  offloaded->body = std::move(offloaded_body);
  offloaded->body->parent_stmt = offloaded;
  offloaded->block_dim = 1;
}

void maybe_convert(OffloadedStmt *stmt, const CompileConfig &config) {
  if ((stmt->task_type == TaskType::range_for) && (!stmt->is_bit_vectorized)) {
    make_multithreaded_range_for(stmt, config);
  }
}

}  // namespace

namespace irpass {

void make_cpu_multithreaded_range_for(IRNode *root,
                                      const CompileConfig &config) {
  if (auto *block = root->cast<Block>()) {
    for (auto &s_ : block->statements) {
      if (auto *s = s_->cast<OffloadedStmt>()) {
        maybe_convert(s, config);
      }
    }
  } else if (auto *s = root->cast<OffloadedStmt>()) {
    maybe_convert(s, config);
  }
  re_id(root);
}

}  // namespace irpass

}  // namespace taichi::lang
