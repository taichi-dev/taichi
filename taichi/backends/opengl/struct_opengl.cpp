#include "struct_opengl.h"

#include "taichi/ir/snode.h"
#include <numeric>

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
  result.root_size = compute_snode_size(node);
  result.snode_map = std::move(snode_map_);
  result.root_snode_type_name = node.node_type_name;
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

size_t OpenglStructCompiler::compute_snode_size(const SNode &snode) {
  if (snode.is_place()) {
    return data_type_size(snode.dt);
  }

  if (snode.type == SNodeType::root) {
    snode_map_.at(snode.node_type_name).mem_offset_in_root = 0;
  }
  size_t ch_size = 0;
  const auto &snode_meta = snode_map_.at(snode.node_type_name);
  size_t acc_alignment_bytes = 0;

  // Sort snode.ch by snode_meta.children_offsets so that we compute
  // the mem_offset_in_root in the right order.
  auto sorted_indices = sort_index_by<size_t>(snode_meta.children_offsets);
  for (size_t i : sorted_indices) {
    auto offset = ch_size + snode_meta.mem_offset_in_root;
    // Pad so that the base address of snode.ch[i] is multiple of its
    // elem_stride.
    auto alignment = snode_map_.at(snode.ch[i]->node_type_name).elem_stride;
    auto alignment_bytes =
        alignment ? alignment - 1 - (offset + alignment - 1) % alignment : 0;
    acc_alignment_bytes += alignment_bytes;
    snode_map_.at(snode.ch[i]->node_type_name).mem_offset_in_root =
        offset + alignment_bytes;
    ch_size += compute_snode_size(*snode.ch[i]);
  }

  int n = snode.num_cells_per_container;
  return acc_alignment_bytes + n * ch_size;
}
}  // namespace opengl
TLANG_NAMESPACE_END
