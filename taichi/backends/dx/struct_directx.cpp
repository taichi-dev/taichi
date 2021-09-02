#include "struct_directx.h"
#include "dx_data_types.h"

TLANG_NAMESPACE_BEGIN

namespace dx {
  StructCompiledResult DxStructCompiler::Run(SNode *node) {
    TI_ASSERT(node->type == SNodeType::root);
    printf(">> CollectSNodes\n");
    CollectSNodes(node);
    printf("<< CollectSNodes\n");

    level = 0;
    TI_TRACE("[DxStructCompiler::Run] {} snodes\n", snodes_.size());

    // Fill SNode info in reverse order
    for (int i = int(snodes_.size()) - 1; i >= 0; i--) {
      SNode *n = snodes_[i];
      GenerateTypes(n);
    }

    StructCompiledResult result;
    result.snode_map = snode_map_;
    result.root_size = ComputeSNodeSize(node);
    return result;
  }

  void DxStructCompiler::CollectSNodes(SNode* node) {
    level++;
    snodes_.push_back(node);

    for (int i = 0; i < std::min(10, level); i++) printf("  ");
    printf("name=%s chunk_size=%d depth=%d n=%d node_type_name=%s\n",
      node->name.c_str(),
      node->chunk_size, node->depth, int(node->n),
      node->node_type_name.c_str());

    for (int i = 0; i<int(node->ch.size()); i++) {
      SNode* ch = node->ch.at(i).get();
      CollectSNodes(ch);
    }
    level--;
  }

  size_t DxStructCompiler::ComputeSNodeSize(SNode *node) {
    TI_TRACE("SNode type name: {}", node->get_node_type_name());

    if (node->is_place()) {
      return data_type_size(node->dt);
    }

    size_t ch_size = 0;
    // ch is vector of unique_ptr
    for (const std::unique_ptr<SNode>& ch : node->ch) {
      ch_size += ComputeSNodeSize(ch.get());
    }

    int n = (node->type == SNodeType::root) ? 1 : int(node->n);
    size_t sz = n * ch_size + dx_get_snode_meta_size(node);
    return sz;
  }

  void DxStructCompiler::GenerateTypes(SNode *snode) {
    const bool is_place = snode->is_place();
    const std::string &node_name = snode->node_type_name;
    const std::string child_name = node_name + "_ch";
    SNodeInfo &snode_info = snode_map_[node_name];
    SNodeInfo &snode_child_info = snode_map_[child_name];

    
  // My debug stuff
    {
      std::stringstream ss;
      ss << "is_place=" << is_place;
      for (auto &&[n, info] : snode_map_) {
        ss << " " << n;
      }
      TI_TRACE(ss.str());
    }

    if (!is_place) {
      size_t stride = 0;
      snode_info.children_offsets.resize(snode->ch.size());
      std::vector<std::pair<int, SNode *>> table;
      for (int i = 0; i < int(snode->ch.size()); ++i) {
        table.emplace_back(i, snode->ch[i].get());
      }
      for (auto &&[i, ch] : table) {
        snode_info.children_offsets[i] = stride;
        stride += snode_map_.at(ch->node_type_name).stride;
      }
      snode_child_info.stride = stride;
    }

    if (is_place) {
      const std::string dt_name = dx_data_type_name(snode->dt);
      snode_info.stride = data_type_size(snode->dt);
    } else if (snode->type == SNodeType::dense ||
               snode->type == SNodeType::dynamic ||
               snode->type == SNodeType::root) {
      const int N = (snode->type == SNodeType::root) ? 1 : snode->n;
      int meta_size = dx_get_snode_meta_size(snode);
      snode_info.length = N;
      snode_info.stride = snode_child_info.stride * N + meta_size;
      snode_info.elem_stride = snode_child_info.stride;
    } else {
      TI_ERROR("SNodeType={} not supported for DX",
               snode_type_name(snode->type));
    }

    // My debug stuff
    {
      std::stringstream ss;
      const int NC = int(snode->ch.size());
      ss << NC << " children, offsets:";
      for (int i = 0; i < NC; i++) {
        if (i > 0)
          ss << ", ";
        ss << snode_info.children_offsets[i];
      }
      TI_TRACE("SNode {}, length={}, stride={}, elem_stride={}\n {}", node_name,
               snode_info.length, snode_info.stride, snode_info.elem_stride, ss.str());
    }
  }
}

TLANG_NAMESPACE_END