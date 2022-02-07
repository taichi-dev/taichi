#include "taichi/ir/ir.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/transforms/utils.h"

TLANG_NAMESPACE_BEGIN

namespace {

void convert_to_range_for(OffloadedStmt *offloaded) {
  TI_ASSERT(offloaded->task_type == OffloadedTaskType::mesh_for);

  DelayedIRModifier modifier;
  auto stmts = irpass::analysis::gather_statements(
      offloaded->body.get(),
      [&](Stmt *stmt) { return stmt->is<MeshIndexConversionStmt>(); });
  for (size_t i = 0; i < stmts.size(); ++i) {
    auto conv_stmt = stmts[i]->cast<MeshIndexConversionStmt>();
    if (conv_stmt->conv_type == mesh::ConvType::l2g) {
      stmts[i]->replace_usages_with(conv_stmt->idx);
      modifier.erase(stmts[i]);
    } else if (conv_stmt->conv_type == mesh::ConvType::l2r) {
      stmts[i]->as<MeshIndexConversionStmt>()->conv_type = mesh::ConvType::g2r;
    }
  }

  modifier.modify_ir();

  offloaded->const_begin = true;
  offloaded->const_end = true;
  offloaded->begin_value = 0;
  offloaded->end_value =
      offloaded->mesh->num_elements.find(offloaded->major_from_type)->second;
  offloaded->mesh = nullptr;
  offloaded->task_type = OffloadedTaskType::range_for;
}

void maybe_convert(OffloadedStmt *offloaded) {
  if (offloaded->task_type == OffloadedTaskType::mesh_for &&
      offloaded->major_to_types.size() == 0) {
    auto stmts = irpass::analysis::gather_statements(  // ti.mesh_patch_idx()
                                                       // relies on mesh-for
        offloaded->body.get(),
        [&](Stmt *stmt) { return stmt->is<MeshPatchIndexStmt>(); });
    if (stmts.size() == 0) {
      convert_to_range_for(offloaded);
    }
  }
}

}  // namespace

namespace irpass {

void demote_no_access_mesh_fors(IRNode *root) {
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

TLANG_NAMESPACE_END
