#include "structural_node.h"
#include "codegen_base.h"

TLANG_NAMESPACE_BEGIN

class StructCompiler : public CodeGenBase {
 public:
  StructCompiler() : CodeGenBase() {
    suffix = "cpp";
    emit_code("#include <common.h>");
    emit_code("using namespace taichi;");
    emit_code("using namespace Tlang;");
    emit_code("\n");
  }

  void visit(SNode &snode) {
    for (auto ch : snode.ch) {
      visit(*ch);
    }

    emit_code("");
    snode.node_type_name = create_snode();
    auto type = snode.type;

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
    emit_code("");

    if (type != SNodeType::place) {
      // Chain accessors for non-leaf nodes
      if (type != SNodeType::forked) {
        // Single child
        auto ch = snode.ch[0];
        emit_code("TC_FORCE_INLINE {} *access_{}({} *parent, int i) {{",
                  ch->node_type_name, ch->node_type_name, snode.node_type_name);
        emit_code("return parent->look_up(i);");
        emit_code("}");
      } else {
        // fork
        for (int i = 0; i < snode.ch.size(); i++) {
          auto ch = snode.ch[i];
          emit_code("TC_FORCE_INLINE {} *access_{}({} *parent, int i) {{",
                    ch->node_type_name, ch->node_type_name, snode.node_type_name);
          emit_code("return parent.get<{}>();", i);
          emit_code("}");
        }
      }
    }
  }

  void generate_leaf_accessors(SNode &snode) {
    // emit end2end accessors for leaf (place) nodes, using chain accessors

  }

  void run(SNode &node) {
    // bottom to top
    visit(node);
    generate_leaf_accessors(node);
    write_code_to_file();
  }
};

TLANG_NAMESPACE_END
