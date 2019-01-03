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
    for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
      auto &ch = snode.ch[ch_id];
      visit(*ch);

      TC_P(ch->type_name());
      int total_bits_start_inferred = ch->total_bit_start + ch->total_num_bits;
      TC_P(ch->total_bit_start);
      TC_P(ch->total_num_bits);
      if (ch_id == 0) {
        snode.total_bit_start = total_bits_start_inferred;
      } else if (snode.parent != nullptr) { // root is ok
        TC_ASSERT(snode.total_bit_start == total_bits_start_inferred);
      }
      // infer extractors
      int acc_offsets = 0;
      for (int i = max_num_indices - 1; i >= 0; i--) {
        int inferred = ch->extractors[i].start + ch->extractors[i].num_bits;
        if (ch_id == 0) {
          snode.extractors[i].start = inferred;
          snode.extractors[i].dest_offset = snode.total_bit_start + acc_offsets;
        } else if (snode.parent != nullptr) { // root is OK
          TC_ASSERT_INFO(snode.extractors[i].start == inferred,
                         "Inconsistent bit configuration");
          TC_ASSERT_INFO(snode.extractors[i].dest_offset ==
                             snode.total_bit_start + acc_offsets,
                         "Inconsistent bit configuration");
        }
        acc_offsets += snode.extractors[i].num_bits;
      }
    }

    snode.total_num_bits = 0;
    for (int i = 0; i < max_num_indices; i++) {
      snode.total_num_bits += snode.extractors[i].num_bits;
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
        for (int i = 0; i < (int)snode.ch.size(); i++) {
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
      emit_code(
          "TLANG_ACCESSOR {} * access_{}(void *root, int i0, int i1=0, int "
          "i2=0, "
          "int i3=0) {{",
          snode.node_type_name, snode.node_type_name);
      emit_code("int tmp;");
      emit_code("auto n0 = ({} *)root;", root_type);
      for (int i = 0; i + 1 < (int)stack.size(); i++) {
        emit_code("tmp = 0;", i);
        for (int j = max_num_indices - 1; j >= 0; j--) {
          auto e = stack[i]->extractors[j];
          int b = e.num_bits;
          if (b) {
            emit_code("tmp = (tmp << {}) + ((i{} >> {}) & ((1 << {}) - 1));",
                      e.num_bits, j, e.start,
                      e.num_bits);  // TODO: 0 should be j
          }
        }
        emit_code("auto n{} = access_{}(n{}, tmp);", i + 1,
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
    emit_code("#if defined(TLANG_KERNEL) ");
    emit_code("#define TLANG_ACCESSOR TC_FORCE_INLINE");
    emit_code("#else");
    emit_code("#define TLANG_ACCESSOR extern \"C\"");
    emit_code("#endif");
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
