#include "opencl_program.h"
#include "opencl_utils.h"

#include "taichi/util/line_appender.h"
#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

// Generate corresponding OpenCL Source Code for Taichi Structures
class OpenclLayoutGen {
 public:
  OpenclLayoutGen(SNode *root) : root(root) {
  }

  std::string compile() {
    TI_ASSERT(root->type == SNodeType::root);
    generate_types(root);

    auto source = line_appender.lines();
    TI_INFO("struct compiled result:\n{}", source);
    return source;
  }

 private:
  void generate_children(SNode *snode) {
    ScopedIndent _s(line_appender);
    for (auto const &ch : snode->ch) {
      generate_types(ch.get());
    }
  }

  void generate_types(SNode *snode) {
    // suffix is for the array size
    auto node_name = snode->node_type_name;
    auto struct_name = snode->get_node_type_name_hinted();

    if (snode->type == SNodeType::place) {
      const auto type = opencl_data_type_name(snode->dt);
      emit("{} {};", type, node_name);

    } else if (snode->type == SNodeType::root) {
      emit("struct Ti_{} {{", struct_name);
      generate_children(snode);
      emit("}};");

    } else if (snode->type == SNodeType::dense) {
      emit("struct Ti_{} {{", struct_name);
      generate_children(snode);
      emit("}} {}[{}];", node_name, snode->n);

    } else {
      TI_ERROR("SNodeType={} not supported on OpenCL backend",
               snode_type_name(snode->type));
    }
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

  SNode *root;
  std::vector<SNode *> snodes;
  LineAppender line_appender;
};

}  // namespace

void OpenclProgram::compile_layout(SNode *root) {
  OpenclLayoutGen gen(root);
  layout_source = gen.compile();
}

}  // namespace opencl
TLANG_NAMESPACE_END
