#include "../ir.h"

TLANG_NAMESPACE_BEGIN

VecStatement convert_to_range_for(StructForStmt *struct_for) {
  VecStatement ret;
  auto loop_var = ret.push_back<AllocaStmt>(DataType::i32);
  auto lower = ret.push_back<ConstStmt>(TypedConstant(0));
  std::vector<SNode *> snodes;
  auto snode = struct_for->snode;
  int total_bits = 0;
  while (snode->type != SNodeType::root) {
    snodes.push_back(snode);
    total_bits += snode->total_num_bits;
  }
  std::reverse(snodes.begin(), snodes.end());
  TC_ASSERT(total_bits <= 31);

  auto upper_bound = 1 << total_bits;
  auto upper = ret.push_back<ConstStmt>(TypedConstant(upper_bound));
  auto body = std::move(struct_for->body);

  auto old_loop_vars = struct_for->loop_vars;
  std::vector<Stmt *> new_loop_vars;

  VecStatement body_header;

  std::vector<int> physical_indices;

  TC_ASSERT(snodes.back()->num_active_indices == (int)old_loop_vars.size());
  for (int i = 0; i < (int)old_loop_vars.size(); i++) {
    new_loop_vars.push_back(body_header.push_back<ConstStmt>(TypedConstant(0)));
    physical_indices.push_back(snodes.back()->physical_index_position[i]);
  }

  auto main_loop_var =
      body_header.push_back<LocalLoadStmt>(LocalAddress(loop_var, 0));

  for (int i = 0; i < (int)snodes.size(); i++) {
    auto snode = snodes[i];
    for (int j = 0; j < (int)physical_indices.size(); j++) {
      auto p = physical_indices[j];
      auto ext = snode->extractors[p];
      auto delta = body_header.push_back<OffsetAndExtractBitsStmt>(
          main_loop_var, ext.start, ext.start + ext.num_bits, 0);
      // TODO: multiply by something?
      new_loop_vars[j] = body_header.push_back<BinaryOpStmt>(
          BinaryOpType::add, new_loop_vars[j], delta);
    }
  }

  for (int i = 0; i < (int)old_loop_vars.size(); i++) {
    // TODO: replace old loop var with the new one
  }

  body->insert(std::move(body_header), 0);

  auto range_for = Stmt::make<RangeForStmt>(
      loop_var, lower, upper, std::move(body), struct_for->vectorize,
      struct_for->parallelize, false);
  ret.push_back(std::move(range_for));
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
    if (auto s = s_->as<StructForStmt>()) {
      auto snode = s->snode;
      bool all_dense = true;
      while (snode->type != SNodeType::root) {
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
}
}  // namespace irpass

TLANG_NAMESPACE_END
