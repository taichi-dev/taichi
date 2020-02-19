#include "struct_opengl.h"

TLANG_NAMESPACE_BEGIN
namespace opengl {

OpenglStructCompiler::CompiledResult OpenglStructCompiler::run(SNode &node)
{
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
// TODO(archibate): really need fit struct_metal so much?
void OpenglStructCompiler::generate_types(const SNode &snode) {
  const bool is_place = snode.is_place();
  if (!is_place) {
    const std::string class_name = snode.node_type_name + "_ch";
    emit("#define {} uint", class_name);
    std::string stride_str;
    for (int i = 0; i < (int)snode.ch.size(); i++) {
      const auto &ch_node_name = snode.ch[i]->node_type_name;
      if (stride_str.empty()) {
        emit("#define {}_get{}(a_) (a_) // {}",
            snode.node_type_name, i, ch_node_name);
        stride_str = ch_node_name + "_stride";
      } else {
        emit("#define {}_get{}(a_) ((a_) + ({})) // {}",
            snode.node_type_name, i, stride_str, ch_node_name);
        stride_str += " + " + ch_node_name + "_stride";
      }
    }
    if (stride_str.empty()) {
      // Is it possible for this to have no children?
      stride_str = "0";
    }
    emit("#define {}_stride ({})", class_name, stride_str);
  }
  emit("");
  const auto &node_name = snode.node_type_name;
  if (is_place) {
    const auto dt_name = opengl_data_type_name(snode.dt);
    emit("#define {} uint // place {}", node_name, dt_name);
    emit("#define {}_stride {} // sizeof({})", node_name, data_type_size(snode.dt), dt_name);
  } else if (snode.type == SNodeType::dense || snode.type == SNodeType::root) {
    emit("#define {} uint // {}", node_name, snode_type_name(snode.type));
    const int n = (snode.type == SNodeType::dense) ? snode.n : 1;
    emit("#define {}_n {}", node_name, n);
    emit("#define {}_stride ({}_ch_stride * {}_n)", node_name, node_name, node_name);
    emit("#define {}_children(a_, i) ((a_) + {}_ch_stride * (i))", node_name, node_name);
  } else {
    TI_ERROR("SNodeType={} not supported on OpenGL",
             snode_type_name(snode.type));
    TI_NOT_IMPLEMENTED;
  }
  emit("");
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
