#include "taichi/struct/struct_metal.h"

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include "taichi/math/arithmetic.h"
#include "taichi/platform/metal/metal_data_types.h"
#include "taichi/platform/metal/metal_kernel_util.h"

TLANG_NAMESPACE_BEGIN
namespace metal {
namespace {
namespace shaders {
#define TI_INSIDE_METAL_CODEGEN
#include "taichi/platform/metal/shaders/runtime_kernels.metal.h"
#include "taichi/platform/metal/shaders/runtime_structs.metal.h"
#include "taichi/platform/metal/shaders/runtime_utils.metal.h"
#undef TI_INSIDE_METAL_CODEGEN

#include "taichi/platform/metal/shaders/runtime_structs.metal.h"

}  // namespace shaders

constexpr size_t kListgenElementSize = sizeof(shaders::ListgenElement);
constexpr size_t kListManagerSize = sizeof(shaders::ListManager);
constexpr size_t kSNodeMetaSize = sizeof(shaders::SNodeMeta);
constexpr size_t kSNodeExtractorsSize = sizeof(shaders::SNodeExtractors);

inline bool is_bitmasked(const SNode &sn) {
  // return (sn.type == SNodeType::dense && sn._bitmasked);
  return (sn.type == SNodeType::dense);
}

inline size_t bitmasks_stride(int n) {
  constexpr int kBitsPerByte = 8;
  const int bytes_needed = iroundup(n, kBitsPerByte) / kBitsPerByte;
  return iroundup(bytes_needed, 8);
}

class StructCompiler {
 public:
  StructCompiledResult run(SNode &root) {
    TI_ASSERT(root.type == SNodeType::root);
    collect_snodes(root);
    // The host side has run this!
    // infer_snode_properties(node);

    auto snodes_rev = snodes_;
    std::reverse(snodes_rev.begin(), snodes_rev.end());
    {
      max_snodes_ = 0;
      for (const auto &sn : snodes_) {
        if (sn->type == SNodeType::root || sn->type == SNodeType::dense) {
          max_snodes_ = std::max(max_snodes_, sn->id);
        }
      }
      ++max_snodes_;
    }

    for (auto &n : snodes_rev) {
      generate_types(*n);
    }
    StructCompiledResult result;
    result.root_size = compute_snode_size(&root, /*num_elems_sofar=*/1);
    result.snode_structs_source_code = std::move(src_code_);
    emit_runtime_structs(&root);
    result.runtime_utils_source_code = std::move(src_code_);
    result.runtime_kernels_source_code = get_runtime_kernels_source_code();
    result.runtime_size = compute_runtime_size();
    result.max_snodes = max_snodes_;
    result.snode_descriptors = std::move(snode_descriptors_);
    TI_INFO("Metal: root_size={} runtime_size={}", result.root_size,
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
               snode.type == SNodeType::root) {
      const bool bitmasked = is_bitmasked(snode);
      const std::string ch_name = fmt::format("{}_ch", node_name);
      emit("struct {} {{", node_name);
      emit("  // {}", snode_type_name(snode.type));
      const int n = (snode.type == SNodeType::dense) ? snode.n : 1;
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

  size_t compute_snode_size(const SNode *sn, int num_elems_sofar) {
    if (sn->is_place()) {
      return metal_data_type_bytes(to_metal_type(sn->dt));
    }

    const int n = (sn->type == SNodeType::dense) ? sn->n : 1;
    num_elems_sofar *= n;
    size_t ch_size = 0;
    for (const auto &ch : sn->ch) {
      const size_t ch_offset = ch_size;
      const auto *ch_sn = ch.get();
      ch_size += compute_snode_size(ch_sn, num_elems_sofar);
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
    if (is_bitmasked(*sn)) {
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
    TI_ASSERT(src_code_.empty());
    src_code_ += shaders::kMetalRuntimeStructsSourceCode;
    src_code_ += "\n";
    src_code_ += shaders::kMetalRuntimeUtilsSourceCode;
    src_code_ += "\n";
    emit("");
    emit("struct Runtime {{");
    emit("  SNodeMeta snode_metas[{}];", max_snodes_);
    emit("  SNodeExtractors snode_extractors[{}];", max_snodes_);
    emit("  ListManager snode_lists[{}];", max_snodes_);
    emit("}};");
  }

  std::string get_runtime_kernels_source_code() const {
    std::stringstream ss;
    ss << shaders::kMetalRuntimeKernelsSourceCode << "\n";
    return ss.str();
  }

  size_t compute_runtime_size() {
    size_t result = (max_snodes_) *
                    (kSNodeMetaSize + kSNodeExtractorsSize + kListManagerSize);
    TI_INFO("Metal runtime fields size: {} bytes", result);
    int total_items = 0;
    for (const auto &kv : snode_descriptors_) {
      total_items += kv.second.total_num_elems_from_root;
    }
    const size_t list_data_size = total_items * kListgenElementSize;
    TI_INFO("Metal runtime list data size: {} bytes", list_data_size);
    result += list_data_size;
    return result;
  }

  void push_indent() { indent_ += "  "; }

  void pop_indent() {
    indent_.pop_back();
    indent_.pop_back();
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    src_code_ += indent_ + fmt::format(f, std::forward<Args>(args)...) + '\n';
  }

  std::vector<SNode *> snodes_;
  int max_snodes_;
  std::string indent_;
  std::string src_code_;
  std::unordered_map<int, SNodeDescriptor> snode_descriptors_;
};

}  // namespace

StructCompiledResult compile_structs(SNode &root) {
  return StructCompiler().run(root);
}
}  // namespace metal
TLANG_NAMESPACE_END
