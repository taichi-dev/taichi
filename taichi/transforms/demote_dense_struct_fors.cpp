#include "taichi/ir/ir.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"

TLANG_NAMESPACE_BEGIN

VecStatement convert_to_range_for(StructForStmt *struct_for) {
  VecStatement ret;
  auto lower = ret.push_back<ConstStmt>(TypedConstant(0));
  std::vector<SNode *> snodes;
  auto snode = struct_for->snode;
  int total_bits = 0;
  while (snode->type != SNodeType::root) {
    snodes.push_back(snode);
    total_bits += snode->total_num_bits;
    snode = snode->parent;
  }
  std::reverse(snodes.begin(), snodes.end());
  TI_ASSERT(total_bits <= 30);

  auto upper_bound = 1 << total_bits;
  auto upper = ret.push_back<ConstStmt>(TypedConstant(upper_bound));
  auto body = std::move(struct_for->body);

  auto num_loop_vars = struct_for->loop_vars.size();
  std::vector<Stmt *> new_loop_vars;

  VecStatement body_header;

  std::vector<int> physical_indices;

  TI_ASSERT(snodes.back()->num_active_indices == (int)num_loop_vars);
  for (int i = 0; i < (int)num_loop_vars; i++) {
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
      Stmt *delta = body_header.push_back<OffsetAndExtractBitsStmt>(
          main_loop_var, ext.acc_offset + offset,
          ext.acc_offset + offset + ext.num_bits, 0);
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

  for (int i = 0; i < (int)num_loop_vars; i++) {
    auto alloca = body_header.push_back<AllocaStmt>(DataType::i32);
    body_header.push_back<LocalStoreStmt>(alloca, new_loop_vars[i]);
    irpass::replace_statements_with(
        body.get(),
        [&](Stmt *s) {
          if (auto loop_index = s->cast<LoopIndexStmt>()) {
            return loop_index->loop == struct_for &&
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

  auto range_for = Stmt::make<RangeForStmt>(
      nullptr, lower, upper, std::move(body), struct_for->vectorize,
      struct_for->parallelize, struct_for->block_dim, false);
  main_loop_var->loop = range_for.get();
  ret.push_back(std::move(range_for));

  // TODO: safe guard range
  return ret;
}

namespace irpass {

void demote_dense_struct_fors(IRNode *root) {
  auto *block = dynamic_cast<Block *>(root);
  std::vector<Stmt *> block_body;
  for (int i = 0; i < (int)block->statements.size(); i++) {
    block_body.push_back(block->statements[i].get());
  }
  for (int i = 0; i < (int)block_body.size(); i++) {
    auto s_ = block_body[i];
    if (auto s = s_->cast<StructForStmt>()) {
      auto snode = s->snode;
      bool all_dense = true;
      while (all_dense && snode->type != SNodeType::root) {
        if (snode->type != SNodeType::dense) {
          all_dense = false;
        }
        snode = snode->parent;
      }
      if (all_dense) {
        s->parent->replace_with(s, convert_to_range_for(s), false);
      }
    }
  }
  re_id(root);
  fix_block_parents(root);
}

}  // namespace irpass

TLANG_NAMESPACE_END
