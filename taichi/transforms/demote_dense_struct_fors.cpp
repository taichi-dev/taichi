#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

namespace {

using TaskType = OffloadedStmt::TaskType;

void convert_to_range_for(OffloadedStmt *offloaded) {
  TI_ASSERT(offloaded->task_type == TaskType::struct_for);

  std::vector<SNode *> snodes;
  auto *snode = offloaded->snode;
  int total_bits = 0;
  while (snode->type != SNodeType::root) {
    snodes.push_back(snode);
    total_bits += snode->total_num_bits;
    snode = snode->parent;
  }
  std::reverse(snodes.begin(), snodes.end());
  TI_ASSERT(total_bits <= 30);

  offloaded->const_begin = true;
  offloaded->const_end = true;
  offloaded->begin_value = 0;
  offloaded->end_value = 1 << total_bits;

  ////// Begin core transformation
  auto body = std::move(offloaded->body);
  const int num_loop_vars =
      snodes.empty() ? 0 : snodes.back()->num_active_indices;

  std::vector<Stmt *> new_loop_vars;

  VecStatement body_header;

  std::vector<int> physical_indices;

  for (int i = 0; i < num_loop_vars; i++) {
    new_loop_vars.push_back(body_header.push_back<ConstStmt>(TypedConstant(0)));
    physical_indices.push_back(snodes.back()->physical_index_position[i]);
  }

  auto main_loop_var = body_header.push_back<LoopIndexStmt>(nullptr, 0);
  // We will set main_loop_var->loop later.

  int offset = total_bits;
  Stmt *test = body_header.push_back<ConstStmt>(TypedConstant(-1));
  bool has_test = false;
  for (int i = 0; i < (int)snodes.size(); i++) {
    auto snode = snodes[i];
    offset -= snode->total_num_bits;
    for (int j = 0; j < (int)physical_indices.size(); j++) {
      auto p = physical_indices[j];
      auto ext = snode->extractors[p];
      Stmt *delta = body_header.push_back<BitExtractStmt>(
          main_loop_var, ext.acc_offset + offset,
          ext.acc_offset + offset + ext.num_bits);
      auto multiplier =
          body_header.push_back<ConstStmt>(TypedConstant(1 << (ext.start)));
      delta = body_header.push_back<BinaryOpStmt>(BinaryOpType::mul, delta,
                                                  multiplier);
      new_loop_vars[j] = body_header.push_back<BinaryOpStmt>(
          BinaryOpType::add, new_loop_vars[j], delta);
    }
  }

  for (int i = 0; i < (int)snodes.size(); i++) {
    auto snode = snodes[i];
    for (int j = 0; j < (int)physical_indices.size(); j++) {
      auto p = physical_indices[j];
      auto num_elements = snode->extractors[p].num_elements
                          << snode->extractors[p].start;
      if (!bit::is_power_of_two(num_elements)) {
        has_test = true;
        auto bound =
            body_header.push_back<ConstStmt>(TypedConstant(num_elements));
        auto cmp = body_header.push_back<BinaryOpStmt>(BinaryOpType::cmp_lt,
                                                       new_loop_vars[j], bound);
        test = body_header.push_back<BinaryOpStmt>(BinaryOpType::bit_and, test,
                                                   cmp);
      }
    }
  }

  for (int i = 0; i < num_loop_vars; i++) {
    auto alloca = body_header.push_back<AllocaStmt>(DataType::i32);
    body_header.push_back<LocalStoreStmt>(alloca, new_loop_vars[i]);
    irpass::replace_statements_with(
        body.get(),
        [&](Stmt *s) {
          if (auto loop_index = s->cast<LoopIndexStmt>()) {
            return loop_index->loop == offloaded &&
                   loop_index->index ==
                       snodes.back()->physical_index_position[i];
          }
          return false;
        },
        [&]() { return Stmt::make<LocalLoadStmt>(LocalAddress(alloca, 0)); });
  }

  if (has_test) {
    // Create an If statement
    auto if_stmt = Stmt::make_typed<IfStmt>(test);
    if_stmt->true_statements = std::move(body);
    body = std::make_unique<Block>();
    body->insert(std::move(if_stmt));
  }
  body->insert(std::move(body_header), 0);

  offloaded->body = std::move(body);
  main_loop_var->loop = offloaded;
  ////// End core transformation

  offloaded->task_type = TaskType::range_for;
}

}  // namespace

namespace irpass {

void demote_dense_struct_fors(IRNode *root) {
  auto *block = dynamic_cast<Block *>(root);
  for (auto &s_ : block->statements) {
    if (auto *s = s_->cast<OffloadedStmt>()) {
      if ((s->task_type == TaskType::struct_for) &&
          s->snode->is_path_all_dense) {
        convert_to_range_for(s);
      }
    }
  }
  re_id(root);
  fix_block_parents(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
