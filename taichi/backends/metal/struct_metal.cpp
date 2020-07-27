#include "struct_metal.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "taichi/backends/metal/constants.h"
#include "taichi/backends/metal/data_types.h"
#include "taichi/backends/metal/features.h"
#include "taichi/backends/metal/kernel_util.h"
#include "taichi/math/arithmetic.h"
#include "taichi/util/line_appender.h"

TLANG_NAMESPACE_BEGIN
namespace metal {
namespace {
namespace shaders {
#define TI_INSIDE_METAL_CODEGEN
#include "taichi/backends/metal/shaders/runtime_structs.metal.h"
#include "taichi/backends/metal/shaders/runtime_utils.metal.h"
#undef TI_INSIDE_METAL_CODEGEN

#include "taichi/backends/metal/shaders/runtime_structs.metal.h"

}  // namespace shaders

constexpr size_t kListManagerDataSize = sizeof(shaders::ListManagerData);
constexpr size_t kSNodeMetaSize = sizeof(shaders::SNodeMeta);
constexpr size_t kSNodeExtractorsSize = sizeof(shaders::SNodeExtractors);

constexpr int kAlignment = 8;

inline size_t bitmasks_stride(int n) {
  constexpr int kBitsPerByte = 8;
  const int bytes_needed = iroundup(n, kBitsPerByte) / kBitsPerByte;
  // The roundup is to align the stride to 8-bytes.
  return iroundup(bytes_needed, kAlignment);
}

inline int get_n(const SNode &sn) {
  // For root, sn.n is 0.
  return sn.type == SNodeType::root ? 1 : sn.n;
}

class StructCompiler {
 public:
  CompiledStructs run(SNode &root) {
    TI_ASSERT(root.type == SNodeType::root);
    collect_snodes(root);
    // The host side has run this!
    // infer_snode_properties(node);

    auto snodes_rev = snodes_;
    std::reverse(snodes_rev.begin(), snodes_rev.end());
    {
      max_snodes_ = 0;
      has_sparse_snode_ = false;
      for (const auto &sn : snodes_) {
        const auto ty = sn->type;
        if (ty == SNodeType::root || ty == SNodeType::dense ||
            ty == SNodeType::bitmasked || ty == SNodeType::dynamic) {
          max_snodes_ = std::max(max_snodes_, sn->id);
        }
        has_sparse_snode_ = has_sparse_snode_ || is_supported_sparse_type(ty);
      }
      ++max_snodes_;
    }

    CompiledStructs result;
    result.root_size = compute_snode_size(&root);
    emit_runtime_structs();
    line_appender_.dump(&result.runtime_utils_source_code);
    result.runtime_size = compute_runtime_size();
    for (auto &n : snodes_rev) {
      generate_types(*n);
    }
    line_appender_.dump(&result.snode_structs_source_code);
    result.need_snode_lists_data = has_sparse_snode_;
    result.max_snodes = max_snodes_;
    result.snode_descriptors = std::move(snode_descriptors_);
    TI_DEBUG("Metal: root_size={} runtime_size={}", result.root_size,
             result.runtime_size);
    return result;
  }

 private:
  void collect_snodes(SNode &snode) {
    snodes_.push_back(&snode);
    for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
      auto &ch = snode.ch[ch_id];
      collect_snodes(*ch);
    }
  }

  void emit_snode_stride(SNodeType ty, const std::string &ch_name, int n) {
    emit("  constant static constexpr int elem_stride = {}::stride;", ch_name);

    if (ty == SNodeType::bitmasked) {
      emit(
          "  constant static constexpr int stride = elem_stride * n + "
          "/*bitmasked=*/{};",
          bitmasks_stride(n));
    } else if (ty == SNodeType::dynamic) {
      emit(
          "  constant static constexpr int stride = elem_stride * n + "
          "/*dynamic=*/{};",
          kAlignment);
    } else {
      // `root`, `dense`
      emit("  constant static constexpr int stride = elem_stride * n;");
    }
    emit("");
  }

  void emit_snode_constructor(const SNode &sn) {
    // TODO(k-ye): handle `pointer`
    const auto ty = sn.type;
    const auto &name = sn.node_type_name;
    if (ty == SNodeType::root) {
      emit("  {}(device byte *addr) {{", name);
    } else {
      emit(
          "  {}(device byte *addr, device Runtime *rtm, device MemoryAllocator "
          "*ma) {{",
          name);
    }
    if (ty == SNodeType::bitmasked || ty == SNodeType::dynamic) {
      emit("    rep_.init(addr, /*meta_offset=*/elem_stride * n);");
    } else {
      // `dense` or `root`
      emit("    rep_.init(addr);");
    }
    emit("  }}\n");
  }

  void emit_snode_get_child_func(SNodeType ty, const std::string &ch_name) {
    // TODO(k-ye): handle `pointer`
    emit("  {} children(int i) {{", ch_name);
    emit("    return {{rep_.addr() + (i * elem_stride)}};");
    emit("  }}\n");
  }

  void emit_snode_activation_funcs(SNodeType ty) {
    emit("  inline bool is_active(int i) {{");
    emit("    return rep_.is_active(i);");
    emit("  }}\n");
    emit("  inline void activate(int i) {{");
    emit("    rep_.activate(i);");
    emit("  }}\n");
    if (ty == SNodeType::dynamic) {
      emit("  inline void deactivate() {{");
      emit("    rep_.deactivate();");
      emit("  }}\n");
    } else {
      emit("  inline void deactivate(int i) {{");
      emit("    rep_.deactivate(i);");
      emit("  }}\n");
    }
  }

  void generate_types(const SNode &snode) {
    const bool is_place = snode.is_place();
    if (!is_place) {
      // Generate {snode}_ch
      const std::string class_name = snode.node_type_name + "_ch";
      emit("class {} {{", class_name);
      emit(" public:");
      emit("  {}(device byte *a) : addr_(a) {{}}", class_name);

      std::string stride_str = "0";
      for (int i = 0; i < (int)snode.ch.size(); i++) {
        const auto &ch_node_name = snode.ch[i]->node_type_name;
        emit("  {} get{}(device Runtime *rtm, device MemoryAllocator *ma) {{",
             ch_node_name, i);
        emit("    return {{addr_ + ({}), rtm, ma}};", stride_str);
        stride_str += " + " + ch_node_name + "::stride";
        emit("  }}");
        emit("");
      }
      emit("  device byte *addr() {{ return addr_; }}");
      emit("");
      // Is it possible for this to have no children?
      emit("  constant static constexpr int stride = {};", stride_str);
      emit(" private:");
      emit("  device byte *addr_;");
      emit("}};");
    }
    emit("");
    const auto &node_name = snode.node_type_name;
    const auto snty = snode.type;
    if (is_place) {
      const auto dt_name = metal_data_type_name(snode.dt);
      emit("struct {} {{", node_name);
      emit("  // place");
      emit("  constant static constexpr int stride = sizeof({});", dt_name);
      emit("");
      // `place` constructor
      emit("  {}(device byte *v, device Runtime *, device MemoryAllocator *)",
           node_name);
      emit("    : val((device {}*)v) {{}}", dt_name);
      emit("");
      emit("  device {}* val;", dt_name);
      emit("}};");
    } else if (snty == SNodeType::dense || snty == SNodeType::root ||
               snty == SNodeType::bitmasked || snty == SNodeType::dynamic) {
      const std::string ch_name = fmt::format("{}_ch", node_name);
      emit("struct {} {{", node_name);
      const auto snty_name = snode_type_name(snty);
      emit("  // {}", snty_name);
      const int n = get_n(snode);
      emit("  constant static constexpr int n = {};", n);
      emit_snode_stride(snty, ch_name, n);
      emit_snode_constructor(snode);
      emit_snode_get_child_func(snty, ch_name);
      emit_snode_activation_funcs(snty);
      if (snty == SNodeType::dynamic) {
        emit("  inline int append(int32_t data) {{");
        emit("    return rep_.append(data);");
        emit("  }}\n");
        emit("  inline int length() {{");
        emit("    return rep_.length();");
        emit("  }}\n");
      }
      emit(" private:");
      emit("  SNodeRep_{} rep_;", snty_name);
      emit("}};");
    } else {
      TI_ERROR("SNodeType={} not supported on Metal", snode_type_name(snty));
      TI_NOT_IMPLEMENTED;
    }
    emit("");
  }

  size_t compute_snode_size(const SNode *sn) {
    if (sn->is_place()) {
      return metal_data_type_bytes(to_metal_type(sn->dt));
    }

    const int n = get_n(*sn);
    size_t ch_size = 0;
    for (const auto &ch : sn->ch) {
      const size_t ch_offset = ch_size;
      const auto *ch_sn = ch.get();
      ch_size += compute_snode_size(ch_sn);
      if (!ch_sn->is_place()) {
        snode_descriptors_.find(ch_sn->id)->second.mem_offset_in_parent =
            ch_offset;
      }
    }
    SNodeDescriptor sn_desc;
    sn_desc.snode = sn;
    sn_desc.element_stride = ch_size;
    sn_desc.num_slots = n;
    sn_desc.stride = ch_size * n;
    if (sn->type == SNodeType::bitmasked) {
      sn_desc.stride += bitmasks_stride(n);
    } else if (sn->type == SNodeType::dynamic) {
      sn_desc.stride += kAlignment;
    }
    sn_desc.total_num_elems_from_root = 1;
    for (const auto &e : sn->extractors) {
      sn_desc.total_num_elems_from_root *= e.num_elements;
    }

    TI_ASSERT(snode_descriptors_.find(sn->id) == snode_descriptors_.end());
    snode_descriptors_[sn->id] = sn_desc;
    return sn_desc.stride;
  }

  void emit_runtime_structs() {
    line_appender_.append_raw(shaders::kMetalRuntimeStructsSourceCode);
    emit("");
    emit("struct Runtime {{");
    emit("  SNodeMeta snode_metas[{}];", max_snodes_);
    emit("  SNodeExtractors snode_extractors[{}];", max_snodes_);
    emit("  ListManagerData snode_lists[{}];", max_snodes_);
    emit("  uint32_t rand_seeds[{}];", kNumRandSeeds);
    emit("}};");
    line_appender_.append_raw(shaders::kMetalRuntimeUtilsSourceCode);
    emit("");
  }

  size_t compute_runtime_size() {
    size_t result = (max_snodes_) * (kSNodeMetaSize + kSNodeExtractorsSize +
                                     kListManagerDataSize);
    result += sizeof(uint32_t) * kNumRandSeeds;
    return result;
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  std::vector<SNode *> snodes_;
  int max_snodes_;
  LineAppender line_appender_;
  std::unordered_map<int, SNodeDescriptor> snode_descriptors_;
  bool has_sparse_snode_;
};

}  // namespace

int SNodeDescriptor::total_num_self_from_root(
    const std::unordered_map<int, SNodeDescriptor> &sn_descs) const {
  if (snode->type == SNodeType::root) {
    return 1;
  }
  const auto *psn = snode->parent;
  TI_ASSERT(psn != nullptr);
  return sn_descs.find(psn->id)->second.total_num_elems_from_root;
}

int total_num_self_from_root(const SNodeDescriptorsMap &m, int snode_id) {
  return m.at(snode_id).total_num_self_from_root(m);
}

CompiledStructs compile_structs(SNode &root) {
  return StructCompiler().run(root);
}
}  // namespace metal
TLANG_NAMESPACE_END
