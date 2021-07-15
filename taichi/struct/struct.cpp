// Codegen for the hierarchical data structure

#include "taichi/ir/ir.h"
#include "taichi/ir/expression.h"
#include "taichi/program/snode_expr_utils.h"
#include "taichi/program/program.h"
#include "struct.h"

namespace taichi {
namespace lang {

void infer_snode_properties(SNode &snode) {
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    ch->parent = &snode;
    for (int i = 0; i < taichi_max_num_indices; i++) {
      ch->extractors[i].num_elements *= snode.extractors[i].num_elements;
      bool found = false;
      for (int k = 0; k < taichi_max_num_indices; k++) {
        if (snode.physical_index_position[k] == i) {
          found = true;
          break;
        }
      }
      if (found)
        continue;
      if (snode.extractors[i].active) {
        snode.physical_index_position[snode.num_active_indices++] = i;
      }
    }

    std::memcpy(ch->physical_index_position, snode.physical_index_position,
                sizeof(snode.physical_index_position));
    ch->num_active_indices = snode.num_active_indices;

    if (snode.type == SNodeType::bit_struct ||
        snode.type == SNodeType::bit_array) {
      ch->is_bit_level = true;
    } else {
      ch->is_bit_level = snode.is_bit_level;
    }

    infer_snode_properties(*ch);
  }

  // infer extractors
  int acc_offsets = 0;
  for (int i = taichi_max_num_indices - 1; i >= 0; i--) {
    snode.extractors[i].acc_offset = acc_offsets;
    acc_offsets += snode.extractors[i].num_bits;
  }
  int acc_shape = 1;
  for (int i = taichi_max_num_indices - 1; i >= 0; i--) {
    snode.extractors[i].acc_shape = acc_shape;
    acc_shape *= snode.extractors[i].shape;
  }
  if (snode.type == SNodeType::dynamic) {
    int active_extractor_counder = 0;
    for (int i = 0; i < taichi_max_num_indices; i++) {
      if (snode.extractors[i].num_bits != 0) {
        active_extractor_counder += 1;
        SNode *p = snode.parent;
        while (p) {
          TI_ASSERT_INFO(
              p->extractors[i].num_bits == 0,
              "Dynamic SNode must have a standalone dimensionality.");
          p = p->parent;
        }
      }
    }
    TI_ASSERT_INFO(active_extractor_counder == 1,
                   "Dynamic SNode can have only one index extractor.");
  }

  snode.total_num_bits = 0;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    snode.total_num_bits += snode.extractors[i].num_bits;
  }
  // The highest bit is for the sign.
  constexpr int kMaxTotalNumBits = 64;
  TI_ERROR_IF(
      snode.total_num_bits >= kMaxTotalNumBits,
      "SNode={}: total_num_bits={} exceeded limit={}. This implies that "
      "your requested shape is too large.",
      snode.id, snode.total_num_bits, kMaxTotalNumBits);

  if (snode.ch.empty()) {
    if (snode.type != SNodeType::place && snode.type != SNodeType::root) {
      TI_ERROR("{} node must have at least one child.",
               snode_type_name(snode.type));
    }
  }

  if (!snode.index_offsets.empty()) {
    TI_ASSERT(snode.index_offsets.size() == snode.num_active_indices);
  }
}

void StructCompiler::collect_snodes(SNode &snode) {
  snodes.push_back(&snode);
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    collect_snodes(*ch);
  }
}

}  // namespace lang
}  // namespace taichi
