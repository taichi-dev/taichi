#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/utils.h"

namespace taichi::lang {

namespace {

using TaskType = OffloadedStmt::TaskType;

void convert_to_range_for(OffloadedStmt *offloaded) {
  TI_ASSERT(offloaded->task_type == TaskType::struct_for);

  std::vector<SNode *> snodes;
  auto *snode = offloaded->snode;
  int64 total_n = 1;
  std::array<int, taichi_max_num_indices> total_shape;
  total_shape.fill(1);
  while (snode->type != SNodeType::root) {
    snodes.push_back(snode);
    for (int j = 0; j < taichi_max_num_indices; j++) {
      total_shape[j] *= snode->extractors[j].shape;
    }
    total_n *= snode->num_cells_per_container;
    snode = snode->parent;
  }
  TI_ASSERT(total_n <= std::numeric_limits<int>::max());
  std::reverse(snodes.begin(), snodes.end());

  offloaded->const_begin = true;
  offloaded->const_end = true;
  offloaded->begin_value = 0;
  offloaded->end_value = total_n;

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

  for (int i = 0; i < (int)snodes.size(); i++) {
    auto snode = snodes[i];
    Stmt *extracted = main_loop_var;
    if (i != 0) {  // first extraction doesn't need a mod
      extracted = generate_mod(&body_header, extracted, total_n);
    }
    total_n /= snode->num_cells_per_container;
    extracted = generate_div(&body_header, extracted, total_n);
    bool is_first_extraction = true;
    for (int j = 0; j < (int)physical_indices.size(); j++) {
      auto p = physical_indices[j];
      auto ext = snode->extractors[p];
      if (!ext.active)
        continue;
      Stmt *index = extracted;
      if (is_first_extraction) {  // first extraction doesn't need a mod
        is_first_extraction = false;
      } else {
        index = generate_mod(&body_header, index, ext.acc_shape * ext.shape);
      }
      index = generate_div(&body_header, index, ext.acc_shape);
      total_shape[p] /= ext.shape;
      auto multiplier =
          body_header.push_back<ConstStmt>(TypedConstant(total_shape[p]));
      auto delta = body_header.push_back<BinaryOpStmt>(BinaryOpType::mul, index,
                                                       multiplier);
      new_loop_vars[j] = body_header.push_back<BinaryOpStmt>(
          BinaryOpType::add, new_loop_vars[j], delta);
    }
  }

  irpass::replace_statements(
      body.get(), /*filter=*/
      [&](Stmt *s) {
        if (auto loop_index = s->cast<LoopIndexStmt>()) {
          return loop_index->loop == offloaded;
        } else {
          return false;
        }
      },
      /*finder=*/
      [&](Stmt *s) {
        auto index = std::find(physical_indices.begin(), physical_indices.end(),
                               s->as<LoopIndexStmt>()->index);
        TI_ASSERT(index != physical_indices.end());
        return new_loop_vars[index - physical_indices.begin()];
      });

  body->insert(std::move(body_header), 0);

  offloaded->body = std::move(body);
  offloaded->body->set_parent_stmt(offloaded);
  main_loop_var->loop = offloaded;
  ////// End core transformation

  offloaded->task_type = TaskType::range_for;
}

void maybe_convert(OffloadedStmt *stmt) {
  if ((stmt->task_type == TaskType::struct_for) &&
      stmt->snode->is_path_all_dense) {
    convert_to_range_for(stmt);
  }
}

}  // namespace

namespace irpass {

void demote_dense_struct_fors(IRNode *root) {
  if (auto *block = root->cast<Block>()) {
    for (auto &s_ : block->statements) {
      if (auto *s = s_->cast<OffloadedStmt>()) {
        maybe_convert(s);
      }
    }
  } else if (auto *s = root->cast<OffloadedStmt>()) {
    maybe_convert(s);
  }
  re_id(root);
}

}  // namespace irpass

}  // namespace taichi::lang
