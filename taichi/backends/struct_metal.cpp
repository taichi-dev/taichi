#include "struct_metal.h"

TLANG_NAMESPACE_BEGIN
namespace metal {

MetalStructCompiler::CompiledResult MetalStructCompiler::run(SNode &node) {
  TI_ASSERT(node.type == SNodeType::root);
  collect_snodes(node);
  // The host side has run this!
  // infer_snode_properties(node);

  auto snodes_rev = snodes_;
  std::reverse(snodes_rev.begin(), snodes_rev.end());

  emit("using byte = uchar;");
  for (auto &n : snodes_rev) {
    generate_types(*n);
  }
  CompiledResult result;
  result.source_code = std::move(src_code_);
  result.root_size = compute_snode_size(node);
  return result;
}

void MetalStructCompiler::collect_snodes(SNode &snode) {
  snodes_.push_back(&snode);
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    collect_snodes(*ch);
  }
}

void MetalStructCompiler::generate_types(const SNode &snode) {
  const bool is_place = snode.is_place();
  if (!is_place) {
    const std::string class_name = snode.node_type_name + "_ch";
    emit("class {} {{", class_name);
    emit(" private:");
    emit("  device byte* addr_;");
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
    }
    if (stride_str.empty()) {
      // Is it possible for this to have no children?
      stride_str = "0";
    }
    emit("  constant static constexpr int stride = {};", stride_str);
    emit("}};");
  }
  emit("");
  const auto &node_name = snode.node_type_name;
  if (is_place) {
    const auto dt_name = metal_data_type_name(snode.dt);
    emit("struct {} {{", node_name);
    emit("  // place");
    emit("  constant static constexpr int stride = sizeof({});", dt_name);
    emit("  {}(device byte* v) : val((device {}*)v) {{}}", node_name, dt_name);
    emit("  device {}* val;", dt_name);
    emit("}};");
  } else if (snode.type == SNodeType::dense || snode.type == SNodeType::root) {
    emit("struct {} {{", node_name);
    emit("  // {}", snode_type_name(snode.type));
    const int n = (snode.type == SNodeType::dense) ? snode.n : 1;
    emit("  constant static constexpr int n = {};", n);
    emit("  constant static constexpr int stride = {}_ch::stride * n;",
         node_name);
    emit("  {}(device byte* a) : addr_(a) {{}}", node_name);
    emit("  {}_ch children(int i) {{", node_name);
    emit("    return {{addr_ + i * {}_ch::stride}};", node_name);
    emit("  }}");
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

size_t MetalStructCompiler::compute_snode_size(const SNode &sn) {
  if (sn.is_place()) {
    return metal_data_type_bytes(to_metal_type(sn.dt));
  }
  size_t ch_size = 0;
  for (const auto &ch : sn.ch) {
    ch_size += compute_snode_size(*ch);
  }
  const int n = (sn.type == SNodeType::dense) ? sn.n : 1;
  return n * ch_size;
}

}  // namespace metal
TLANG_NAMESPACE_END
