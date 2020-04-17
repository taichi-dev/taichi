#include "struct_opengl.h"

TLANG_NAMESPACE_BEGIN
namespace opengl {

OpenglStructCompiler::CompiledResult OpenglStructCompiler::run(SNode &node) {
  TI_ASSERT(node.type == SNodeType::root);
  collect_snodes(node);
  // The host side has run this!
  // infer_snode_properties(node);

  auto snodes_rev = snodes_;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  for (auto &n : snodes_rev) {
    generate_types(*n);
  }
  CompiledResult result;
  result.source_code = std::move(src_code_);
  result.class_get_map = std::move(class_get_map_);
  result.class_children_map = std::move(class_children_map_);
  result.root_size = compute_snode_size(node);
  return result;
}

void OpenglStructCompiler::collect_snodes(SNode &snode) {
  snodes_.push_back(&snode);
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    collect_snodes(*ch);
  }
}

void OpenglStructCompiler::generate_types(const SNode &snode) {
  const bool is_place = snode.is_place();
  if (!is_place) {
    const std::string class_name = snode.node_type_name + "_ch";
    emit("#define {} const int", class_name);
    std::string stride_str;
    size_t stride_num = 0;
    auto &gimme = class_get_map_.at(snode.node_type_name);
    gimme.clear();
    for (int i = 0; i < (int)snode.ch.size(); i++) {
      const auto &ch_node_name = snode.ch[i]->node_type_name;
      if (stride_str.empty()) {
        emit("#define {}_get{}(a_) (a_) // {}", snode.node_type_name, i,
             ch_node_name);
        gimme.push_back(0);
        stride_str = ch_node_name + "_stride";
        stride_num = stride_map_.at(ch_node_name);
      } else {
        emit("#define {}_get{}(a_) ((a_) + {}) // {}", snode.node_type_name, i,
             stride_num, ch_node_name);
        gimme.push_back(stride_num);
        stride_str += " + " + ch_node_name + "_stride";
        stride_num += stride_map_.at(ch_node_name);
      }
    }
    if (stride_str.empty()) {
      // Is it possible for this to have no children?
      stride_str = "0";
    }
    emit("#define {}_stride {}", class_name, stride_num);
    stride_map_[class_name] = stride_num;
  }
  const auto &node_name = snode.node_type_name;
  if (is_place) {
    const auto dt_name = opengl_data_type_name(snode.dt);
    emit("#define {} const int // place {}", node_name, dt_name);
    emit("#define {}_stride {} // sizeof({})", node_name,
         data_type_size(snode.dt), dt_name);
    stride_map_[node_name] = data_type_size(snode.dt);
  } else if (snode.type == SNodeType::dense || snode.type == SNodeType::root) {
    emit("#define {} const int // {}", node_name, snode_type_name(snode.type));
    const int n = (snode.type == SNodeType::dense) ? snode.n : 1;
    emit("#define {}_n {}", node_name, n);
    length_map_[node_name] = n;
    emit("#define {}_stride ({}_ch_stride * {}_n)", node_name, node_name,
         node_name);
    stride_map_[node_name] =
        stride_map_[node_name + "_ch"] * length_map_[node_name];
    emit("#define {}_children(a_, i) ((a_) + {}_ch_stride * (i))", node_name,
         node_name);
    class_children_map_[node_name] = stride_map_.at(node_name + "_ch");
  } else {
    TI_ERROR("SNodeType={} not supported on OpenGL",
             snode_type_name(snode.type));
    TI_NOT_IMPLEMENTED;
  }
  TI_INFO("SSRC:\n{}", src_code_);
}

size_t OpenglStructCompiler::compute_snode_size(const SNode &sn) {
  if (sn.is_place()) {
    return data_type_size(sn.dt);
  }
  size_t ch_size = 0;
  for (const auto &ch : sn.ch) {
    ch_size += compute_snode_size(*ch);
  }
  const int n = (sn.type == SNodeType::dense) ? sn.n : 1;
  return n * ch_size;
}

}  // namespace opengl
TLANG_NAMESPACE_END
