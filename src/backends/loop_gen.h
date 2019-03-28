#pragma once
#include "base.h"

TLANG_NAMESPACE_BEGIN

class LoopGenerator {
 public:
  CodeGenBase *gen;

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    gen->emit(f, std::forward<Args>(args)...);
  }

  LoopGenerator(CodeGenBase *gen) : gen(gen) {
  }

  std::string loop_variable(SNode *snode) {
    return snode->node_type_name + "_loop";
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

  void single_loop_body_head(SNode *snode) {
    if (snode->parent == nullptr) {
      emit("auto {}_cache = root;", snode->node_type_name);
    } else {
      auto parent = fmt::format("{}_cache", snode->parent->node_type_name);
      emit("auto {}_cache = access_{}({}, {});", snode->node_type_name,
           snode->node_type_name, parent, loop_variable(snode->parent));
    }
  }

  void generate_single_loop_header(SNode *snode, bool leaf = false) {
    if (!leaf)
      single_loop_body_head(snode);

    auto l = loop_variable(snode);
    if (snode->type == SNodeType::pointer) {
      emit("if (!{}_cache->data) continue;", snode->node_type_name, l);
    }
    if (snode->type != SNodeType::hashed) {
      emit("int {};", l);
      emit("auto {}_cache_n = {}_cache->get_n();", snode->node_type_name,
           snode->node_type_name);
    }
    if (snode->_multi_threaded) {
      TC_NOT_IMPLEMENTED
    }

    if (snode->type == SNodeType::hashed) {
      emit("for (auto &{}_it : {}_cache->data) {{", l, snode->node_type_name);
      emit("int {} = {}_it.first;", l, l);
    } else {
      emit("for ({} = 0; {} < {}_cache_n; {} += {}) {{", l, l,
           snode->node_type_name, l, 1);
    }

    update_indices(snode);
  }

  void update_indices(SNode *snode) {
    // update indices....
    auto l = loop_variable(snode);
    for (int i = 0; i < max_num_indices; i++) {
      std::string ancester = "0 |";
      if (snode->parent != nullptr) {
        ancester = index_name_global(snode->parent, i) + " |";
      }
      std::string addition = "0";
      if (snode->extractors[i].num_bits) {
        addition = fmt::format(
            "((({} >> {}) & ((1 << {}) - 1)) << {})", l,
            snode->extractors[i].dest_offset - snode->total_bit_start,
            snode->extractors[i].num_bits, snode->extractors[i].start);
      }
      emit("int {} = {};", index_name_local(snode, i), addition);
      emit("int {} = {} {};", index_name_global(snode, i), ancester,
           index_name_local(snode, i));
    }
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
    emit("LeafContext<{}> leaf_context;", leaf->node_type_name);
    single_loop_body_head(leaf);
    emit("leaf_context.ptr = {}_cache;", leaf->node_type_name);
    for (int i = 0; i < max_num_indices; i++)
      emit("leaf_context.indices[{}] = {};", i,
           index_name_global(leaf->parent, i));
    emit("leaves.push_back(leaf_context);");
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
        if (for_stmt->snode->physical_index_position[i] == j)
          emit("auto {} = {};", for_stmt->loop_vars[i]->raw_name(),
               index_name_global(leaf, j));
      }
    }
  }

};

TLANG_NAMESPACE_END
