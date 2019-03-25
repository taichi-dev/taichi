#include "../tlang.h"
#include "struct.h"

TLANG_NAMESPACE_BEGIN

StructCompiler::StructCompiler() : CodeGenBase() {
  snode_count = 0;
  creator = nullptr;
  suffix = "cpp";
  emit("#define TLANG_HOST");
  emit("#include <common.h>");
  emit("using namespace taichi;");
  emit("using namespace Tlang;");
  emit("\n");
}

void StructCompiler::visit(SNode &snode) {
  // TC_P(snode.type_name());
  for (int ch_id = 0; ch_id < (int)snode.ch.size(); ch_id++) {
    auto &ch = snode.ch[ch_id];
    for (int i = 0; i < max_num_indices; i++) {
      bool found = false;
      for (int k = 0; k < max_num_indices; k++) {
        if (snode.index_order[k] == i) {
          found = true;
          break;
        }
      }
      if (found)
        continue;
      if (snode.extractors[i].num_bits) {
        snode.index_order[snode.num_active_indices++] = i;
      }
    }
    /*
    TC_TAG;
    for (int i = 0; i < max_num_indices; i++) {
      fmt::print("{}, ", snode.index_order[i]);
    }
    fmt::print("\n");
    */
    std::memcpy(ch->index_order, snode.index_order, sizeof(snode.index_order));
    ch->num_active_indices = snode.num_active_indices;
    visit(*ch);

    // TC_P(ch->type_name());
    int total_bits_start_inferred = ch->total_bit_start + ch->total_num_bits;
    // TC_P(ch->total_bit_start);
    // TC_P(ch->total_num_bits);
    if (ch_id == 0) {
      snode.total_bit_start = total_bits_start_inferred;
    } else if (snode.parent != nullptr) {  // root is ok
      TC_ASSERT(snode.total_bit_start == total_bits_start_inferred);
    }
    // infer extractors
    int acc_offsets = 0;
    for (int i = max_num_indices - 1; i >= 0; i--) {
      int inferred = ch->extractors[i].start + ch->extractors[i].num_bits;
      if (ch_id == 0) {
        snode.extractors[i].start = inferred;
        snode.extractors[i].dest_offset = snode.total_bit_start + acc_offsets;
      } else if (snode.parent != nullptr) {  // root is OK
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

  emit("");
  snode.node_type_name = create_snode();
  auto type = snode.type;

  if (snode.type != SNodeType::indirect && snode.type != SNodeType::place &&
      snode.ch.empty()) {
    TC_ERROR("Non-place node should have at least one child.");
  }

  if (snode.has_ambient) {
    emit("{} {}_ambient = {};", snode.data_type_name(), snode.node_type_name,
         snode.ambient_val.stringify());
  }

  // create children type that supports forking...
  emit("struct {}_ch {{", snode.node_type_name);
  emit("static constexpr int n=1;");
  for (int i = 0; i < (int)snode.ch.size(); i++) {
    emit("{} member{};", snode.ch[i]->node_type_name, i);
  }
  for (int i = 0; i < (int)snode.ch.size(); i++) {
    emit("TC_DEVICE {} *get{}() {{return &member{};}} ",
         snode.ch[i]->node_type_name, i, i);
  }
  // emit("TC_FORCE_INLINE int get_n() {{return 1;}} ");
  emit("}};");

  if (type == SNodeType::fixed) {
    emit("using {} = fixed<{}_ch, {}>;", snode.node_type_name,
         snode.node_type_name, snode.n);
  } else if (type == SNodeType::root) {
    emit("using {} = layout_root<{}_ch>;", snode.node_type_name,
         snode.node_type_name);
  } else if (type == SNodeType::dynamic) {
    emit("using {} = dynamic<{}_ch, {}>;", snode.node_type_name,
         snode.node_type_name, snode.n);
  } else if (type == SNodeType::indirect) {
    emit("using {} = indirect<{}_ch>;", snode.node_type_name, snode.n);
  } else if (type == SNodeType::pointer) {
    emit("using {} = pointer<{}_ch>;", snode.node_type_name,
         snode.node_type_name);
  } else if (type == SNodeType::hashed) {
    emit("using {} = hashed<{}_ch>;", snode.node_type_name,
         snode.node_type_name);
  } else if (type == SNodeType::place) {
    emit("using {} = {};", snode.node_type_name, snode.data_type_name());
  } else {
    TC_P(snode.type_name());
    TC_NOT_IMPLEMENTED;
  }
  emit("");
}

void StructCompiler::generate_leaf_accessors(SNode &snode) {
  auto type = snode.type;
  stack.push_back(&snode);

  bool is_leaf = type == SNodeType::place || type == SNodeType ::indirect;

  if (!is_leaf) {
    // Chain accessors for non-leaf nodes
    TC_ASSERT(snode.ch.size() > 0);
    for (int i = 0; i < (int)snode.ch.size(); i++) {
      auto ch = snode.ch[i];
      emit("TLANG_ACCESSOR {} *access_{}({} *parent, int i) {{",
           ch->node_type_name, ch->node_type_name, snode.node_type_name);
      emit(
          "auto lookup = parent->look_up(i); "
          "if ({}::has_null && lookup == nullptr) return nullptr;",
          snode.node_type_name);
      emit("return lookup->get{}();", i);
      emit("}}");
    }
    emit("");
  }
  // SNode::place & indirect
  // emit end2end accessors for leaf (place) nodes, using chain accessors
  emit(
      "TLANG_ACCESSOR TC_EXPORT {} * access_{}(void *root, int i0, int i1=0, "
      "int "
      "i2=0, "
      "int i3=0) {{",
      snode.node_type_name, snode.node_type_name);
  if (snode._verbose) {
    emit(
        "std::cout << \"accessing node {} at \" << i0 << ' ' << i1 << ' ' "
        "<< i2 << ' ' << i3 << std::endl;",
        snode.node_type_name);
  }
  emit("int tmp;");
  emit("auto n0 = ({} *)root;", root_type);
  for (int i = 0; i + 1 < (int)stack.size(); i++) {
    emit("tmp = 0;", i);
    for (int j = 0; j < max_num_indices; j++) {
      auto e = stack[i]->extractors[j];
      int b = e.num_bits;
      if (b) {
        if (e.num_bits == e.start || max_num_indices != 1) {
          emit("tmp = (tmp << {}) + ((i{} >> {}) & ((1 << {}) - 1));",
               e.num_bits, j, e.start, e.num_bits);
        } else {
          TC_WARN("Emitting shortcut indexing");
          emit("tmp = i{};", j);
        }
      }
    }
    emit("auto n{} = access_{}(n{}, tmp);", i + 1, stack[i + 1]->node_type_name,
         i);
    if (snode.has_ambient && stack[i + 1] != &snode)
      emit("if ({}::has_null && n{} == nullptr) return &{}_ambient;",
           stack[i]->node_type_name, i + 1, snode.node_type_name);
  }
  emit("return n{};", (int)stack.size() - 1);
  emit("}}");
  emit("");

  if (type == SNodeType::indirect) {  // toucher
    emit(
        "TLANG_ACCESSOR void touch_{}(void *root, int val, int i0, int "
        "i1=0, int "
        "i2=0, "
        "int i3=0) {{",
        snode.node_type_name);
    emit("auto node = access_{}(root, i0, i1, i2, i3);", snode.node_type_name);
    emit("node->touch(val);");
    emit("}");
  }

  if (type == SNodeType::dynamic) {  // toucher
    emit(
        "TLANG_ACCESSOR void touch_{}(void *root, const {}::child_type &val, "
        "int i0, int "
        "i1=0, int "
        "i2=0, "
        "int i3=0) {{",
        snode.node_type_name, snode.node_type_name);
    emit("auto node = access_{}(root, i0, i1, i2, i3);", snode.node_type_name);
    emit("node->touch(val);");
    emit("}}");
  }

  for (auto ch : snode.ch) {
    generate_leaf_accessors(*ch);
  }

  stack.pop_back();
}

void StructCompiler::load_accessors(SNode &snode) {
  for (auto ch : snode.ch) {
    load_accessors(*ch);
  }
  if (snode.type == SNodeType::place) {
    snode.func = load_function<SNode::AccessorFunction>(
        fmt::format("access_{}", snode.node_type_name));
  }
}

void StructCompiler::set_parents(SNode &snode) {
  for (auto &c : snode.ch) {
    set_parents(*c);
    c->parent = &snode;
  }
}

void StructCompiler::run(SNode &node) {
  set_parents(node);
  // bottom to top
  visit(node);
  root_type = node.node_type_name;
  generate_leaf_accessors(node);
  emit(
      "TC_EXPORT void *create_data_structure() {{auto addr = "
      "taichi::Tlang::allocator()->alloc(sizeof({})); auto p= new(addr) {}; ",
      root_type, root_type);
  /*
  for (int i = 0; i < (int)node.ch.size(); i++) {
    if (node.ch[i]->type != SNodeType::hashed) {
      emit("std::memset(p->children.get{}(), 0, sizeof({}));", i,
           node.ch[i]->node_type_name);
    }
  }
  */
  emit("return p;}}");
  emit(
      "TC_EXPORT void release_data_structure(void *ds) {{delete ({} "
      "*)ds;}}",
      root_type);
  write_source();

  auto cmd = get_current_program().config.compile_cmd(get_source_path(),
                                                      get_library_path());
  auto compile_ret = std::system(cmd.c_str());

  if (compile_ret != 0) {
    auto cmd = get_current_program().config.compile_cmd(
        get_source_path(), get_library_path(), true);
    trash(std::system(cmd.c_str()));
    TC_ERROR("Compilation failed");
  }
  disassemble();
  load_dll();
  creator = load_function<void *(*)()>("create_data_structure");
  load_accessors(node);
}

TLANG_NAMESPACE_END
