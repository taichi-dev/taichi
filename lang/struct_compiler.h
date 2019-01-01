#include "structural_node.h"
#include "codegen_base.h"

TLANG_NAMESPACE_BEGIN

class StructCompiler : public CodeGenBase {
 public:
  std::vector<SNode *> stack;
  std::string root_type;
  void *(*creator)();

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
  }

  void generate_leaf_accessors(SNode &snode) {
    auto type = snode.type;
    stack.push_back(&snode);

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
                    ch->node_type_name, ch->node_type_name,
                    snode.node_type_name);
          emit_code("return parent->get<{}>();", i);
          emit_code("}");
        }
      }
      emit_code("");
    } else {
      // emit end2end accessors for leaf (place) nodes, using chain accessors
      emit_code("extern \"C\" {} * access_{}(void *root, int i) {{",
                snode.node_type_name, snode.node_type_name);
      emit_code("auto n0 = ({} *)root;", root_type);
      for (int i = 0; i + 1 < stack.size(); i++) {
        emit_code("auto n{} = access_{}(n{},i);", i + 1,
                  stack[i + 1]->node_type_name, i);
      }
      emit_code("return n{};", (int)stack.size() - 1);
      emit_code("}");
      emit_code("");
    }

    for (auto ch : snode.ch) {
      generate_leaf_accessors(*ch);
    }

    stack.pop_back();
  }

  void load_accessors(SNode &snode) {
    for (auto ch : snode.ch) {
      load_accessors(*ch);
    }
    if (snode.type == SNodeType::place) {
      snode.func = load_function<SNode::AccessorFunction>(
          fmt::format("access_{}", snode.node_type_name));
    }
  }

  void run(SNode &node) {
    // bottom to top
    visit(node);
    root_type = node.node_type_name;
    generate_leaf_accessors(node);
    emit_code("extern \"C\" void *create_data_structure() {{return new {};}}",
              root_type);
    emit_code(
        "extern \"C\" void release_data_structure(void *ds) {{delete ({} "
        "*)ds;}}",
        root_type);
    write_code_to_file();

    auto cmd = fmt::format(
        "g++ {} -std=c++14 -shared -fPIC -O3 -march=native -I {}/headers -Wall "
        "-D_GLIBCXX_USE_CXX11_ABI=0 -DTLANG_CPU -o {}",
        get_source_fn(), get_project_fn(), get_library_fn());
    auto compile_ret = std::system(cmd.c_str());
    TC_ASSERT(compile_ret == 0);
    disassemble();
    load_dll();
    creator = load_function<void *(*)()>("create_data_structure");
    load_accessors(node);
  }
};

TLANG_NAMESPACE_END
