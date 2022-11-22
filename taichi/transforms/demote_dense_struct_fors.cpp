#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/utils.h"

namespace taichi::lang {

namespace {

using TaskType = OffloadedStmt::TaskType;

void convert_to_range_for(OffloadedStmt *offloaded, bool packed) {
  TI_ASSERT(offloaded->task_type == TaskType::struct_for);

  std::vector<SNode *> snodes;
  auto *snode = offloaded->snode;
  int total_bits = 0;
  int start_bits[taichi_max_num_indices] = {0};
  while (snode->type != SNodeType::root) {
    snodes.push_back(snode);
    for (int j = 0; j < taichi_max_num_indices; j++) {
      start_bits[j] += snode->extractors[j].num_bits;
    }
    total_bits += snode->total_num_bits;
    snode = snode->parent;
  }
  std::reverse(snodes.begin(), snodes.end());

  // general shape calculation - no dependence on POT
  int64 total_n = 1;
  std::array<int, taichi_max_num_indices> total_shape;
  total_shape.fill(1);
  for (const auto *s : snodes) {
    for (int j = 0; j < taichi_max_num_indices; j++) {
      total_shape[j] *= s->extractors[j].shape;
    }
    total_n *= s->num_cells_per_container;
  }
  TI_ASSERT(total_n <= std::numeric_limits<int>::max());

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

  Stmt *test = body_header.push_back<ConstStmt>(TypedConstant(-1));
  bool has_test = false;
  if (packed) {  // no dependence on POT
    for (int i = 0; i < (int)snodes.size(); i++) {
      auto snode = snodes[i];
      auto old_total_n = total_n;
      total_n /= snode->num_cells_per_container;
      auto extracted =
          i == 0 ? main_loop_var
                 : generate_mod(
                       &body_header, main_loop_var,
                       old_total_n);  // first extraction doesn't need a mod
      extracted = generate_div(&body_header, extracted, total_n);
      for (int j = 0; j < (int)physical_indices.size(); j++) {
        auto p = physical_indices[j];
        auto ext = snode->extractors[p];
        auto index =
            j == 0 ? extracted
                   : generate_mod(
                         &body_header, extracted,
                         ext.acc_shape *
                             ext.shape);  // first extraction doesn't need a mod
        index = generate_div(&body_header, index, ext.acc_shape);
        total_shape[p] /= ext.shape;
        auto multiplier =
            body_header.push_back<ConstStmt>(TypedConstant(total_shape[p]));
        auto delta = body_header.push_back<BinaryOpStmt>(BinaryOpType::mul,
                                                         index, multiplier);
        new_loop_vars[j] = body_header.push_back<BinaryOpStmt>(
            BinaryOpType::add, new_loop_vars[j], delta);
      }
    }
  } else {
    int offset = total_bits;
    for (int i = 0; i < (int)snodes.size(); i++) {
      auto snode = snodes[i];
      offset -= snode->total_num_bits;
      for (int j = 0; j < (int)physical_indices.size(); j++) {
        auto p = physical_indices[j];
        auto ext = snode->extractors[p];
        Stmt *delta = body_header.push_back<BitExtractStmt>(
            main_loop_var, ext.acc_offset + offset,
            ext.acc_offset + offset + ext.num_bits);
        start_bits[p] -= ext.num_bits;
        auto multiplier =
            body_header.push_back<ConstStmt>(TypedConstant(1 << start_bits[p]));
        delta = body_header.push_back<BinaryOpStmt>(BinaryOpType::mul, delta,
                                                    multiplier);
        new_loop_vars[j] = body_header.push_back<BinaryOpStmt>(
            BinaryOpType::add, new_loop_vars[j], delta);
      }
    }

    if (!snodes.empty()) {
      auto snode = snodes.back();
      for (int j = 0; j < (int)physical_indices.size(); j++) {
        auto p = physical_indices[j];
        auto num_elements = snode->extractors[p].num_elements_from_root;
        if (!bit::is_power_of_two(num_elements)) {
          has_test = true;
          auto bound =
              body_header.push_back<ConstStmt>(TypedConstant(num_elements));
          auto cmp = body_header.push_back<BinaryOpStmt>(
              BinaryOpType::cmp_lt, new_loop_vars[j], bound);
          test = body_header.push_back<BinaryOpStmt>(BinaryOpType::bit_and,
                                                     test, cmp);
        }
      }
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

  if (has_test) {
    // Create an If statement
    auto if_stmt = Stmt::make_typed<IfStmt>(test);
    if_stmt->set_true_statements(std::move(body));
    // Note that this could silently change the body block of |offloaded|.
    body = std::make_unique<Block>();
    body->insert(std::move(if_stmt));
  }
  body->insert(std::move(body_header), 0);

  offloaded->body = std::move(body);
  offloaded->body->parent_stmt = offloaded;
  main_loop_var->loop = offloaded;
  ////// End core transformation

  offloaded->task_type = TaskType::range_for;
}

void maybe_convert(OffloadedStmt *stmt, bool packed) {
  if ((stmt->task_type == TaskType::struct_for) &&
      stmt->snode->is_path_all_dense) {
    convert_to_range_for(stmt, packed);
  }
}

}  // namespace

namespace irpass {

void demote_dense_struct_fors(IRNode *root, bool packed) {
  if (auto *block = root->cast<Block>()) {
    for (auto &s_ : block->statements) {
      if (auto *s = s_->cast<OffloadedStmt>()) {
        maybe_convert(s, packed);
      }
    }
  } else if (auto *s = root->cast<OffloadedStmt>()) {
    maybe_convert(s, packed);
  }
  re_id(root);
}

}  // namespace irpass

}  // namespace taichi::lang
