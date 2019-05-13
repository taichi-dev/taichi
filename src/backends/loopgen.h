#pragma once
#include "base.h"

TLANG_NAMESPACE_BEGIN

class LoopGenerator {
 public:
  CodeGenBase *gen;
  int grid_dim;

  LoopGenerator(CodeGenBase *gen);

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    gen->emit(f, std::forward<Args>(args)...);
  }

  std::string loop_variable(SNode *snode) {
    return snode->node_type_name + "_loop";
  }

  std::string index_variable(SNode *snode) {
    return snode->node_type_name + "_index";
  }

  std::string index_name_local(SNode *snode, int i) {
    return fmt::format("index_{}_{}_local", snode->node_type_name, i);
  }

  std::string index_name_global(SNode *snode, int i) {
    return fmt::format("index_{}_{}_global", snode->node_type_name, i);
  }

  void generate_loop_header(SNode *snode, StructForStmt *stmt) {
    TC_ASSERT(snode->type != SNodeType::place)
    if (snode->parent != nullptr) {
      generate_loop_header(snode->parent, stmt);
    }
    generate_single_loop_header(snode);
  }

  // load the SNode cache, given loop variables and a pointer to its parent
  void single_loop_body_head(SNode *snode) {
    if (snode->parent == nullptr) {
      emit("auto {}_cache = root;", snode->node_type_name);
    } else {
      auto parent = fmt::format("{}_cache", snode->parent->node_type_name);
      emit("auto {}_cache = access_{}({}, {});", snode->node_type_name,
           snode->node_type_name, parent, index_variable(snode->parent));
    }
  }

  // compute the physical indices of snode, based on the physical indices of it
  // parent and loop var
  void update_indices(SNode *snode) {
    auto l = loop_variable(snode);
    int morton_id = 0;
    for (int i = 0; i < max_num_indices; i++) {
      std::string ancestor = "0 |";
      if (snode->parent != nullptr) {
        ancestor = index_name_global(snode->parent, i) + " |";
      }
      std::string addition = "0";
      uint32 mask_base[3]{0xffffffff, 0x55555555, 0x49249249};
      if (snode->extractors[i].num_bits) {
        if (!snode->_morton) {
          // no reorder
          addition = fmt::format("((({} >> {}) & ((1 << {}) - 1)) << {})", l,
                                 snode->extractors[i].acc_offset,
                                 snode->extractors[i].num_bits,
                                 snode->extractors[i].start);
        } else {
          TC_ASSERT(snode->num_active_indices <= 3);
          uint32 mask = mask_base[snode->num_active_indices - 1]
                        << (snode->num_active_indices - morton_id - 1);
          morton_id++;
          addition = fmt::format("(_pext_u32({}, {}) << {})", l, mask,
                                 snode->extractors[i].start);
        }
      }
      emit("int {} = {};", index_name_local(snode, i), addition);
      emit("int {} = {} {};", index_name_global(snode, i), ancestor,
           index_name_local(snode, i));
    }
    if (snode->_morton) {
      emit("int {} = 0;", index_variable(snode));
      for (int i = 0; i < max_num_indices; i++) {
        emit("{} = {} | {};", index_variable(snode), index_variable(snode),
             index_name_local(snode, i));
      }
    } else {
      emit("int {} = {};", index_variable(snode), l);
    }
  }

  void generate_single_loop_header(SNode *snode,
                                   bool leaf = false,
                                   int step_size = 1) {
    if (!leaf)
      single_loop_body_head(snode);

    auto l = loop_variable(snode);
    if (snode->type == SNodeType::pointer) {
      emit("if (!{}_cache->data) continue;", snode->node_type_name, l);
    }
    if (snode->type != SNodeType::hash || true) {
      emit("int {};", l);
      emit("auto {}_cache_n = {}_cache->get_n();", snode->node_type_name,
           snode->node_type_name);
    }
    if (snode->_multi_threaded) {
      TC_NOT_IMPLEMENTED
    }

    if (snode->type == SNodeType::hash) {
      // emit("for (auto &{}_it : {}_cache->data) {{", l,
      // snode->node_type_name); emit("int {} = {}_it.first;", l, l);
      emit("for (int {}_e=0;{}_e < {}_cache_n; {}_e++) {{", l, l,
           snode->node_type_name, l);
      emit("int {} = {}_cache->entries[{}_e];", l, snode->node_type_name, l);
    } else {
      emit("for ({} = 0; {} < {}_cache_n; {} += {}) {{", l, l,
           snode->node_type_name, l, step_size);
    }
    if (snode->need_activation()) {
      emit("if (!{}_cache->is_active({})) continue;", snode->node_type_name, l);
    }

    update_indices(snode);
  }

  void generate_loop_tail(SNode *snode, StructForStmt *stmt) {
    emit("}}\n");
    if (snode->parent != nullptr) {
      generate_loop_tail(snode->parent, stmt);
    } else {
      return;  // no loop for root, which is a fork
    }
  }

  void loop_gen_leaves(StructForStmt *for_stmt, SNode *leaf) {
    emit("std::vector<LeafContext<{}>> leaves;", leaf->node_type_name);
    generate_loop_header(leaf->parent, for_stmt);
    single_loop_body_head(leaf);
    if (leaf->type == SNodeType::dynamic) {
      emit("if ({}_cache && {}_cache->get_n())", leaf->node_type_name,
           leaf->node_type_name);
    }
    if (leaf->type == SNodeType::pointer) {
      emit("if ({}_cache->data)", leaf->node_type_name, leaf->node_type_name);
    }
    emit("{{");
    emit("LeafContext<{}> leaf_context;", leaf->node_type_name);
    emit("leaf_context.ptr = {}_cache;", leaf->node_type_name);
    for (int i = 0; i < max_num_indices; i++)
      emit("leaf_context.indices[{}] = {};", i,
           index_name_global(leaf->parent, i));
    emit("leaves.push_back(leaf_context);");
    emit("}}");
    generate_loop_tail(leaf->parent, for_stmt);
  }

  void emit_load_from_context(SNode *leaf) {
    emit("auto {}_cache = leaves[leaf_loop].ptr;", leaf->node_type_name);
    for (int i = 0; i < max_num_indices; i++) {
      emit("auto {} = leaves[leaf_loop].indices[{}];",
           index_name_global(leaf->parent, i), i);
    }
  }

  void emit_setup_loop_variables(StructForStmt *for_stmt, SNode *leaf) {
    for (int i = 0; i < (int)for_stmt->loop_vars.size(); i++) {
      for (int j = 0; j < max_num_indices; j++) {
        if (for_stmt->snode->physical_index_position[i] == j) {
          emit("{} {} = {};", for_stmt->loop_vars[i]->ret_data_type_name(),
               for_stmt->loop_vars[i]->raw_name(), index_name_global(leaf, j));
          if (leaf->parent) {
            emit("auto {}_base = {};", for_stmt->loop_vars[i]->raw_name(),
                 index_name_global(leaf->parent, j));
          }
        }
      }
    }
  }

  void emit_listgen_func(SNode *snode,
                         int child_block_division = 0,
                         std::string suffix = "");

  std::string listgen_func_name(SNode *leaf, std::string suffix = "") {
    return fmt::format("{}_listgen{}", leaf->node_type_name, suffix);
  }
};

TLANG_NAMESPACE_END
