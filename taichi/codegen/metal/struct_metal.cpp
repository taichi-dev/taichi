#include "struct_metal.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "taichi/rhi/metal/constants.h"
#include "taichi/runtime/metal/data_types.h"
#include "taichi/runtime/metal/features.h"
#include "taichi/runtime/metal/kernel_utils.h"
#include "taichi/math/arithmetic.h"
#include "taichi/util/line_appender.h"

namespace taichi {
namespace lang {
namespace metal {
namespace {
namespace shaders {
#define TI_INSIDE_METAL_CODEGEN
#include "taichi/runtime/metal/shaders/runtime_structs.metal.h"
#include "taichi/runtime/metal/shaders/runtime_utils.metal.h"
#include "taichi/runtime/metal/shaders/snode_bit_pointer.metal.h"
#undef TI_INSIDE_METAL_CODEGEN

#include "taichi/runtime/metal/shaders/runtime_structs.metal.h"

}  // namespace shaders

constexpr size_t kListManagerDataSize = sizeof(shaders::ListManagerData);
constexpr size_t kNodeManagerDataSize = sizeof(shaders::NodeManagerData);
constexpr size_t kNodeManagerElemIndexSize =
    sizeof(shaders::NodeManagerData::ElemIndex);
constexpr size_t kSNodeMetaSize = sizeof(shaders::SNodeMeta);
constexpr size_t kSNodeExtractorsSize = sizeof(shaders::SNodeExtractors);

static_assert(kNodeManagerElemIndexSize == sizeof(int32_t),
              "sizeof(NodeManagerData::ElemIndex) != 4");

constexpr int kAlignment = 8;

inline size_t bitmasks_stride(int n) {
  constexpr int kBitsPerByte = 8;
  const int bytes_needed = iroundup(n, kBitsPerByte) / kBitsPerByte;
  // The roundup is to align the stride to 8-bytes.
  return iroundup(bytes_needed, kAlignment);
}

class StructCompiler {
 public:
  CompiledStructs run(SNode &root) {
    TI_ASSERT(root.type == SNodeType::root);
    collect_snodes(root);
    auto snodes_rev = snodes_;
    std::reverse(snodes_rev.begin(), snodes_rev.end());
    {
      max_snodes_ = 0;
      has_sparse_snode_ = false;
#define CHECK_UNSUPPORTED_TYPE(type_case)                                \
  else if (ty == SNodeType::type_case) {                                 \
    TI_ERROR("Metal backend does not support SNode=" #type_case " yet"); \
  }
      for (const auto &sn : snodes_) {
        const auto ty = sn->type;
        if (ty == SNodeType::place) {
          // do nothing
        }
        CHECK_UNSUPPORTED_TYPE(quant_array)
        CHECK_UNSUPPORTED_TYPE(hash)
        else {
          max_snodes_ = std::max(max_snodes_, sn->id);
        }
        has_sparse_snode_ = has_sparse_snode_ || is_supported_sparse_type(ty);
      }
      ++max_snodes_;
    }
#undef CHECK_UNSUPPORTED_TYPE

    CompiledStructs result;
    result.root_snode_type_name = root.node_type_name;
    result.root_size = compute_snode_size(&root);
    for (auto &n : snodes_rev) {
      generate_types(*n);
    }
    line_appender_.dump(&result.snode_structs_source_code);
    result.root_id = root.id;
    result.max_snodes = max_snodes_;
    result.snode_descriptors = std::move(snode_descriptors_);
    TI_DEBUG("Metal: root_id={} root_size={}", result.root_id,
             result.root_size);
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
    if (ty == SNodeType::pointer) {
      emit(
          "  constant static constexpr int elem_stride = "
          "/*pointer ElemIndex=*/{};",
          kNodeManagerElemIndexSize);
    } else {
      emit("  constant static constexpr int elem_stride = {}::stride;",
           ch_name);
    }

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
      // `root`, `dense`, `bit_struct`
      emit("  constant static constexpr int stride = elem_stride * n;");
    }
    emit("");
  }

  void emit_snode_constructor(const SNode &sn) {
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
    } else if (ty == SNodeType::pointer) {
      const auto snid = sn.id;
      emit("    NodeManager nm;");
      emit("    nm.nm_data = (rtm->snode_allocators + {});", snid);
      emit("    nm.mem_alloc = ma;");
      emit("    const auto amb_idx = rtm->ambient_indices[{}];", snid);
      emit("    rep_.init(addr, nm, amb_idx);");
    } else if (ty == SNodeType::root || ty == SNodeType::dense) {
      // `root`, `dense`
      emit("    rep_.init(addr);");
    } else {
      TI_UNREACHABLE;
    }
    emit("  }}\n");
  }

  void emit_snode_get_child_func(SNodeType ty, const std::string &ch_name) {
    emit("  {} children(int i) {{", ch_name);
    if (ty == SNodeType::pointer) {
      emit("    return {{rep_.child_or_ambient_addr(i)}};");
    } else {
      emit("    return {{rep_.addr() + (i * elem_stride)}};");
    }
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
    if (snode.is_bit_level) {
      // Nothing to generate for bit-level SNodes -- they are part of their
      // parent's intrinsic memory.
      return;
    }
    const auto snty = snode.type;
    const bool is_place = snode.is_place();
    const bool should_gen_cell = !(is_place || (snty == SNodeType::bit_struct));
    if (should_gen_cell) {
      // "_ch" is a legacy word for child. The correct notion should be cell.
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
      emit("  device {} *val;", dt_name);
      emit("}};");
    } else if (snty == SNodeType::bit_struct) {
      // TODO: bit_struct and place share a lot in common.
      const auto dt_name = metal_data_type_name(DataType(snode.physical_type));
      emit("struct {} {{", node_name);
      emit("  // bit_struct");
      emit("  constant static constexpr int stride = sizeof({});", dt_name);
      emit("");
      // `bit_struct` constructor
      emit("  {}(device byte *b, device Runtime *, device MemoryAllocator *)",
           node_name);
      emit("    : base(b) {{}}");
      emit("");
      emit("  device byte *base;");
      emit("}};");
    } else if (snty == SNodeType::dense || snty == SNodeType::root ||
               snty == SNodeType::bitmasked || snty == SNodeType::dynamic ||
               snty == SNodeType::pointer) {
      const std::string ch_name = fmt::format("{}_ch", node_name);
      emit("struct {} {{", node_name);
      const auto snty_name = snode_type_name(snty);
      emit("  // {}", snty_name);
      const int64 n = snode.num_cells_per_container;
      // There's no assert in metal shading language yet so we have to warn
      // outside.
      if (n > std::numeric_limits<int>::max()) {
        TI_WARN(
            "{}: Snode index might be out of int32 boundary but int64 is not "
            "supported on metal backend.",
            node_name);
      }
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
      // We have checked the type support previously.
      TI_UNREACHABLE;
    }
    emit("");
  }

  size_t compute_snode_size(const SNode *sn) {
    if (sn->is_place()) {
      return metal_data_type_bytes(to_metal_type(sn->dt));
    }
    if (sn->is_bit_level) {
      // A bit-level SNode occupies a fration of a byte. Just return 0 here and
      // special handling the bit_* SNode containers.
      return 0;
    }
    const int n = sn->num_cells_per_container;
    size_t ch_size = 0;
    if (sn->type == SNodeType::bit_struct) {
      // The host side should have inferred all the necessary info of |sn|.
      TI_ASSERT(sn->physical_type != nullptr);
      ch_size = data_type_size(sn->physical_type);
      // |ch_size| should at least be 4 bytes on GPU. In addition, Metal:
      // 1. does not support 8-byte data types in the device address space.
      // 2. only supports 4-byte atomic integral types (or atomic_bool).
      TI_ERROR_IF(ch_size != 4,
                  "bit_struct physical type must be exactly 32 bits on Metal");
    } else {
      for (const auto &ch : sn->ch) {
        const size_t ch_offset = ch_size;
        const auto *ch_sn = ch.get();
        ch_size += compute_snode_size(ch_sn);
        if (!ch_sn->is_place()) {
          snode_descriptors_.find(ch_sn->id)->second.mem_offset_in_parent =
              ch_offset;
        }
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
    } else if (sn->type == SNodeType::pointer) {
      // A `pointer` SNode itself only stores pointers, not the actual data!
      sn_desc.stride = n * kNodeManagerElemIndexSize;
    }
    sn_desc.total_num_elems_from_root = 1;
    for (const auto &e : sn->extractors) {
      sn_desc.total_num_elems_from_root *= e.num_elements_from_root;
    }

    TI_ASSERT(snode_descriptors_.find(sn->id) == snode_descriptors_.end());
    snode_descriptors_[sn->id] = sn_desc;
    return sn_desc.stride;
  }

  template <typename... Args>
  void emit(std::string f, Args &&...args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  std::vector<SNode *> snodes_;
  int max_snodes_{0};
  LineAppender line_appender_;
  std::unordered_map<int, SNodeDescriptor> snode_descriptors_;
  bool has_sparse_snode_{false};
};

class RuntimeModuleCompiler {
 public:
  CompiledRuntimeModule run() {
    CompiledRuntimeModule res;
    emit_runtime_structs();
    line_appender_.dump(&res.runtime_utils_source_code);
    res.rand_seeds_size = compute_rand_seeds_size();
    res.runtime_size = compute_snodes_runtime_size() + res.rand_seeds_size;
    return res;
  }

 private:
  void emit_runtime_structs() {
    line_appender_.append_raw(shaders::kMetalRuntimeStructsSourceCode);
    emit("");
    emit("struct Runtime {{");
    emit("  uint32_t rand_seeds[{}];", kNumRandSeeds);
    emit("  SNodeMeta snode_metas[{}];", kMaxNumSNodes);
    emit("  SNodeExtractors snode_extractors[{}];", kMaxNumSNodes);
    emit("  ListManagerData snode_lists[{}];", kMaxNumSNodes);
    emit("  NodeManagerData snode_allocators[{}];", kMaxNumSNodes);
    emit("  NodeManagerData::ElemIndex ambient_indices[{}];", kMaxNumSNodes);
    emit("}};");
    emit("");
    line_appender_.append_raw(shaders::kMetalRuntimeUtilsSourceCode);
    emit("");
    line_appender_.append_raw(shaders::kMetalSNodeBitPointerSourceCode);
    emit("");
  }

  size_t compute_snodes_runtime_size() {
    return kMaxNumSNodes *
           (kSNodeMetaSize + kSNodeExtractorsSize + kListManagerDataSize +
            kNodeManagerDataSize + kNodeManagerElemIndexSize);
  }

  size_t compute_rand_seeds_size() {
    return sizeof(uint32_t) * kNumRandSeeds;
  }

  template <typename... Args>
  void emit(std::string f, Args &&...args) {
    line_appender_.append(std::move(f), std::move(args)...);
  }

  LineAppender line_appender_;
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

CompiledRuntimeModule compile_runtime_module() {
  return RuntimeModuleCompiler{}.run();
}

}  // namespace metal
}  // namespace lang
}  // namespace taichi
