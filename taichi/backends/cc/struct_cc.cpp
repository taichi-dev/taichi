#include "struct_cc.h"
#include "cc_layout.h"
#include "cc_utils.h"

TLANG_NAMESPACE_BEGIN
namespace cccp {

void CCLayoutGen::generate_children(SNode *snode) {
  ScopedIndent _s(line_appender);
  for (auto const &ch : snode->ch) {
    generate_types(ch.get());
  }
}

void CCLayoutGen::generate_types(SNode *snode) {
  // suffix is for the array size
  auto node_name = snode->node_type_name;
  auto struct_name = "Ti_" + snode->get_node_type_name_hinted();

  if (snode->type == SNodeType::place) {
    const auto type = cc_data_type_name(snode->dt);
    emit("{} {};", type, node_name);

  } else if (snode->type == SNodeType::root) {
    emit("struct {} {{", struct_name);
    generate_children(snode);
    emit("}};");

  } else if (snode->type == SNodeType::dense) {
    emit("struct {} {{", struct_name);
    generate_children(snode);
    emit("}} {}[{}];", node_name, snode->num_cells_per_container);

  } else {
    TI_ERROR("SNodeType={} not supported on C backend",
             snode_type_name(snode->type));
  }
}

std::unique_ptr<CCLayout> CCLayoutGen::compile() {
  TI_ASSERT(root->type == SNodeType::root);
  generate_types(root);

  auto lay = std::make_unique<CCLayout>(program);
  lay->source = line_appender.lines();
  return lay;
}

}  // namespace cccp
TLANG_NAMESPACE_END
