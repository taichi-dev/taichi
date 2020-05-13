#include "struct_metal.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "taichi/backends/metal/constants.h"
#include "taichi/backends/metal/data_types.h"
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

constexpr size_t kListgenElementSize = sizeof(shaders::ListgenElement);
constexpr size_t kListManagerSize = sizeof(shaders::ListManager);
constexpr size_t kSNodeMetaSize = sizeof(shaders::SNodeMeta);
constexpr size_t kSNodeExtractorsSize = sizeof(shaders::SNodeExtractors);

inline size_t bitmasks_stride(int n) {
  constexpr int kBitsPerByte = 8;
  const int bytes_needed = iroundup(n, kBitsPerByte) / kBitsPerByte;
  // The roundup is to align the stride to 8-bytes.
  return iroundup(bytes_needed, 8);
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
            ty == SNodeType::bitmasked) {
          max_snodes_ = std::max(max_snodes_, sn->id);
        }
        has_sparse_snode_ = has_sparse_snode_ || (ty == SNodeType::bitmasked);
      }
      ++max_snodes_;
    }

    for (auto &n : snodes_rev) {
      generate_types(*n);
    }
    CompiledStructs result;
    result.root_size = compute_snode_size(&root);
    line_appender_.dump(&result.snode_structs_source_code);
    emit_runtime_structs(&root);
    line_appender_.dump(&result.runtime_utils_source_code);
    result.runtime_size = compute_runtime_size();
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

  void generate_types(const SNode &snode) {
    const bool is_place = snode.is_place();
    if (!is_place) {
      const std::string class_name = snode.node_type_name + "_ch";
      emit("class {} {{", class_name);
      emit(" public:");
      emit("  {}(device byte* a) : addr_(a) {{}}", class_name);

      std::string stride_str;
      for (int i = 0; i < (int)snode.ch.size(); i++) {
        const auto &ch_node_name = snode.ch[i]->node_type_name;
        emit("  {} get{}() {{", ch_node_name, i);
        if (stride_str.empty()) {
          emit("    return {{addr_}};");
          stride_str = ch_node_name + "::stride";
        } else {
          emit("    return {{addr_ + ({})}};", stride_str);
          stride_str += " + " + ch_node_name + "::stride";
        }
        emit("  }}");
        emit("");
      }
      emit("  device byte* addr() {{ return addr_; }}");
      emit("");
      if (stride_str.empty()) {
        // Is it possible for this to have no children?
        stride_str = "0";
      }
      emit("  constant static constexpr int stride = {};", stride_str);
      emit(" private:");
      emit("  device byte* addr_;");
      emit("}};");
    }
    emit("");
    const auto &node_name = snode.node_type_name;
    if (is_place) {
      const auto dt_name = metal_data_type_name(snode.dt);
      emit("struct {} {{", node_name);
      emit("  // place");
      emit("  constant static constexpr int stride = sizeof({});", dt_name);
      emit("  {}(device byte* v) : val((device {}*)v) {{}}", node_name,
           dt_name);
      emit("  device {}* val;", dt_name);
      emit("}};");
    } else if (snode.type == SNodeType::dense ||
               snode.type == SNodeType::root ||
               snode.type == SNodeType::bitmasked) {
      const bool bitmasked = snode.type == SNodeType::bitmasked;
      const std::string ch_name = fmt::format("{}_ch", node_name);
      emit("struct {} {{", node_name);
      emit("  // {}", snode_type_name(snode.type));
      const int n = get_n(snode);
      emit("  constant static constexpr int n = {};", n);
      if (bitmasked) {
        emit(
            "  constant static constexpr int stride = {}::stride * n + "
            "/*bitmasks=*/{};",
            ch_name, bitmasks_stride(n));
      } else {
        emit("  constant static constexpr int stride = {}::stride * n;",
             ch_name);
      }
      emit("  {}(device byte* a) : addr_(a) {{}}", node_name);
      emit("");
      emit("  {} children(int i) {{", ch_name);
      emit("    return {{addr_ + i * {}::stride}};", ch_name);
      emit("  }}");
      emit("");
      emit(" private:");
      emit("  device byte* addr_;");
      emit("}};");
    } else {
      TI_ERROR("SNodeType={} not supported on Metal",
               snode_type_name(snode.type));
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
    }
    sn_desc.total_num_elems_from_root = 1;
    for (const auto &e : sn->extractors) {
      sn_desc.total_num_elems_from_root *= e.num_elements;
    }

    TI_ASSERT(snode_descriptors_.find(sn->id) == snode_descriptors_.end());
    snode_descriptors_[sn->id] = sn_desc;
    return sn_desc.stride;
  }

  void emit_runtime_structs(const SNode *root) {
    line_appender_.append_raw(shaders::kMetalRuntimeStructsSourceCode);
    emit("");
    line_appender_.append_raw(shaders::kMetalRuntimeUtilsSourceCode);
    emit("");
    emit("struct Runtime {{");
    emit("  SNodeMeta snode_metas[{}];", max_snodes_);
    emit("  SNodeExtractors snode_extractors[{}];", max_snodes_);
    emit("  ListManager snode_lists[{}];", max_snodes_);
    emit("  uint32_t rand_seeds[{}];", kNumRandSeeds);
    emit("}};");
  }

  size_t compute_runtime_size() {
    size_t result = (max_snodes_) *
                    (kSNodeMetaSize + kSNodeExtractorsSize + kListManagerSize);
    result += sizeof(uint32_t) * kNumRandSeeds;
    TI_DEBUG("Metal runtime fields size: {} bytes", result);
    if (has_sparse_snode_) {
      // We only need additional memory to hold sparsity information. Don't
      // allocate it if there is no sparse SNode at all.
      int total_items = 0;
      for (const auto &kv : snode_descriptors_) {
        total_items += kv.second.total_num_elems_from_root;
      }
      const size_t list_data_size = total_items * kListgenElementSize;
      TI_DEBUG("Metal runtime sparse list data size: {} bytes", list_data_size);
      result += list_data_size;
    }
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

CompiledStructs compile_structs(SNode &root) {
  return StructCompiler().run(root);
}
}  // namespace metal
TLANG_NAMESPACE_END
