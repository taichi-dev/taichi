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
    size_t stride_num = 0;
    auto &gimme = class_get_map_[snode.node_type_name];
    gimme.resize(snode.ch.size());
    std::vector<std::pair<int, SNode *>> smp;
    for (int i = 0; i < (int)snode.ch.size(); i++) {
      smp.push_back(std::pair<int, SNode *>(i, snode.ch[i].get()));
    }
    // Discussion: https://github.com/taichi-dev/taichi/issues/804
    // The NVIDIA OpenGL seems to stuck at compile (a very long time to compile)
    // when a small SNode is placed at a very high address... And I've no idea why.
    // So let's sort by SNode size, smaller first:
    std::sort(smp.begin(), smp.end(),
              [this](const std::pair<int, SNode *> &a,
                     const std::pair<int, SNode *> &b) {
                return stride_map_.at(a.second->node_type_name) <
                       stride_map_.at(b.second->node_type_name);
              });
    for (auto &&[i, ch] : smp) {
      gimme[i] = stride_num;
      stride_num += stride_map_.at(ch->node_type_name);
    }
    stride_map_[class_name] = stride_num;
  }
  const auto &node_name = snode.node_type_name;
  if (is_place) {
    const auto dt_name = opengl_data_type_name(snode.dt);
    stride_map_[node_name] = data_type_size(snode.dt);
  } else if (snode.type == SNodeType::dense || snode.type == SNodeType::root) {
    const int n = (snode.type == SNodeType::dense) ? snode.n : 1;
    length_map_[node_name] = n;
    stride_map_[node_name] =
        stride_map_[node_name + "_ch"] * length_map_[node_name];
    class_children_map_[node_name] = stride_map_.at(node_name + "_ch");
  } else {
    TI_ERROR("SNodeType={} not supported on OpenGL",
             snode_type_name(snode.type));
    TI_NOT_IMPLEMENTED;
  }
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
