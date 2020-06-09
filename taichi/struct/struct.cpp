// Codegen for the hierarchical data structure

#include "taichi/ir/ir.h"
#include "taichi/ir/expression.h"
#include "taichi/program/program.h"
#include "struct.h"

TLANG_NAMESPACE_BEGIN

StructCompiler::StructCompiler(Program *prog) : prog(prog) {
  root_size = 0;
}

void StructCompiler::collect_snodes(SNode &snode) {
  snodes.push_back(&snode);
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    collect_snodes(*ch);
  }
}

void StructCompiler::infer_snode_properties(SNode &snode) {
  // TI_P(snode.type_name());
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
    infer_snode_properties(*ch);

    int total_bits_start_inferred = ch->total_bit_start + ch->total_num_bits;
    if (ch_id == 0) {
      snode.total_bit_start = total_bits_start_inferred;
    } else if (snode.parent != nullptr) {  // root is ok
      // TI_ASSERT(snode.total_bit_start == total_bits_start_inferred);
    }
    // infer extractors
    int acc_offsets = 0;
    for (int i = taichi_max_num_indices - 1; i >= 0; i--) {
      int inferred = ch->extractors[i].start + ch->extractors[i].num_bits;
      if (ch_id == 0) {
        snode.extractors[i].start = inferred;
        snode.extractors[i].acc_offset = acc_offsets;
      } else if (snode.parent != nullptr) {  // root is OK
        /*
        TI_ASSERT_INFO(snode.extractors[i].start == inferred,
                       "Inconsistent bit configuration");
        TI_ASSERT_INFO(snode.extractors[i].dest_offset ==
                           snode.total_bit_start + acc_offsets,
                       "Inconsistent bit configuration");
                       */
      }
      acc_offsets += snode.extractors[i].num_bits;
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
  }

  if (snode.expr.expr)
    snode.expr->set_attribute("dim", std::to_string(snode.num_active_indices));

  snode.total_num_bits = 0;
  for (int i = 0; i < taichi_max_num_indices; i++) {
    snode.total_num_bits += snode.extractors[i].num_bits;
  }

  if (snode.has_null()) {
    ambient_snodes.push_back(&snode);
  }

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

TLANG_NAMESPACE_END
