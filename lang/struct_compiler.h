#include "structural_node.h"
#include "codegen_base.h"

TLANG_NAMESPACE_BEGIN

class StructCompiler : public CodeGenBase {
 public:
  StructCompiler() : CodeGenBase() {
    suffix = ".cpp";
    emit_code("#include <common.h>");
  }

  void visit(SNode &snode) {
    for (auto ch : snode.ch) {
      visit(*ch);
    }
    snode.node_type_name = create_snode();
    auto type = snode.type;
    TC_P(snode.type_name());
    if (type == SNodeType::fixed) {
      emit_code("using {} = fixed<{}, {}>;", snode.node_type_name,
                snode.ch[0]->node_type_name, snode.n);
    } else if (type == SNodeType::forked) {
      std::vector<std::string> c;
      for (auto ch : snode.ch) {
        c.push_back(ch->node_type_name);
      }
      emit_code("using {} = forked{};", snode.node_type_name,
                vec_to_list(c, "<"));
    } else if (type == SNodeType::place) {
      emit_code("using {} = {};", snode.node_type_name,
                snode.addr->data_type_name());
    } else {
      TC_P(snode.type_name());
      TC_NOT_IMPLEMENTED;
    }
  }

  void run(SNode &node) {
    // bottom to top
    visit(node);
  }
};

TLANG_NAMESPACE_END
