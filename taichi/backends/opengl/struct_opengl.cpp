#include "struct_opengl.h"

#include "taichi/ir/snode.h"
#include <numeric>

TLANG_NAMESPACE_BEGIN
namespace opengl {

OpenglStructCompiler::CompiledResult OpenglStructCompiler::run(SNode &node) {
  TI_ASSERT(node.type == SNodeType::root);

  generate_snode_tree(node);

  CompiledResult result;
  result.root_size = snode_map_.at(node.node_type_name).stride;
  result.snode_map = std::move(snode_map_);
  result.root_snode_type_name = node.node_type_name;
  return result;
}

void OpenglStructCompiler::generate_snode_tree(const SNode &root) {
  collect_snodes(root);
  // The host side has run this!
  // infer_snode_properties(node);

  for (int i = snodes_.size() - 1; i >= 0; i--) {
    generate_types(*snodes_[i]);
  }
  snode_map_.at(root.node_type_name).mem_offset_in_root = 0;
  align_as_elem_stride(root);
}

void OpenglStructCompiler::collect_snodes(const SNode &snode) {
  snodes_.push_back(&snode);
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    collect_snodes(*ch);
  }
}

void OpenglStructCompiler::generate_types(const SNode &snode) {
  const bool is_place = snode.is_place();
  const auto &node_name = snode.node_type_name;
  const auto child_name = node_name + "_ch";
  auto &snode_info = snode_map_[node_name];
  snode_info.snode = &snode;
  SNodeInfo snode_child_info;
  if (!is_place) {
    size_t stride_num = 0;
    snode_info.children_offsets.resize(snode.ch.size());
    std::vector<std::pair<int, SNode *>> table;
    for (int i = 0; i < (int)snode.ch.size(); i++) {
      table.push_back(std::pair<int, SNode *>(i, snode.ch[i].get()));
    }
    // Discussion: https://github.com/taichi-dev/taichi/issues/804
    // The NVIDIA OpenGL seems to stuck at compile (a very long time to compile)
    // when a small SNode is placed at a very high address... And I've no idea
    // why. So let's sort by SNode size, smaller first:
    std::sort(table.begin(), table.end(),
              [this](const std::pair<int, SNode *> &a,
                     const std::pair<int, SNode *> &b) {
                return snode_map_[a.second->node_type_name].stride <
                       snode_map_[b.second->node_type_name].stride;
              });
    for (auto &&[i, ch] : table) {
      snode_info.children_offsets[i] = stride_num;
      stride_num += snode_map_.at(ch->node_type_name).stride;
    }
    snode_child_info.stride = stride_num;
  }
  if (is_place) {
    const auto dt_name = opengl_data_type_name(snode.dt);
    snode_info.stride = data_type_size(snode.dt);
  } else if (snode.type == SNodeType::dense || snode.type == SNodeType::root) {
    int64 n = snode.num_cells_per_container;
    snode_info.length = n;
    snode_info.stride = snode_child_info.stride * n;   // my stride
    snode_info.elem_stride = snode_child_info.stride;  // my child stride
  } else {
    TI_ERROR(
        "SNodeType={} not supported on OpenGL\n"
        "Consider use ti.init(ti.cpu) or ti.init(ti.cuda) if you "
        "want to use sparse data structures",
        snode_type_name(snode.type));
    TI_NOT_IMPLEMENTED;
  }
}

namespace {
template <typename T>
std::vector<size_t> sort_index_by(const std::vector<T> &v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
            [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return idx;
}
}  // namespace

void OpenglStructCompiler::align_as_elem_stride(const SNode &snode) {
  size_t ch_size = 0;
  auto &snode_meta = snode_map_.at(snode.node_type_name);
  if (snode.is_place()) {
    ch_size = data_type_size(snode.dt);
  } else {
    // Sort snode.ch by snode_meta.children_offsets so that we compute
    // the mem_offset_in_root in the right order.
    auto sorted_indices = sort_index_by<size_t>(snode_meta.children_offsets);
    for (size_t i : sorted_indices) {
      auto offset = ch_size + snode_meta.mem_offset_in_root;
      // Pad so that the base address of snode.ch[i] is multiple of its
      // elem_stride.
      auto &ch_snode_meta = snode_map_.at(snode.ch[i]->node_type_name);
      auto alignment = ch_snode_meta.elem_stride;
      auto alignment_bytes =
          alignment ? alignment - 1 - (offset + alignment - 1) % alignment : 0;
      auto ch_mem_offset_in_root = offset + alignment_bytes;
      ch_snode_meta.mem_offset_in_root = ch_mem_offset_in_root;
      snode_meta.children_offsets[i] =
          ch_mem_offset_in_root - snode_meta.mem_offset_in_root;

      align_as_elem_stride(*snode.ch[i]);
      ch_size += (alignment_bytes + ch_snode_meta.stride);
    }
  }
  snode_meta.elem_stride = ch_size;
  snode_meta.stride = snode.num_cells_per_container * ch_size;
}
}  // namespace opengl
TLANG_NAMESPACE_END
