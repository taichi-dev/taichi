#include "opencl_program.h"
#include "opencl_utils.h"

#include "taichi/util/line_appender.h"
#include "taichi/ir/snode.h"

#include <tuple>

TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

// Generate corresponding OpenCL Source Code for Taichi Structures
class OpenclLayoutGen {
 public:
  OpenclLayoutGen(SNode *root) : root(root) {
  }

  std::tuple<std::string, size_t> compile() {
    TI_ASSERT(root->type == SNodeType::root);
    size_t size = generate_types(root);

    auto source = line_appender.lines();
    TI_INFO("root buffer (size {}):\n{}", size, source);
    return std::make_tuple(source, size);
  }

 private:
  size_t generate_children(SNode *snode) {
    size_t size = 0;
    ScopedIndent _s(line_appender);
    for (auto const &ch : snode->ch) {
      size += generate_types(ch.get());
    }
    return size;
  }

  size_t generate_types(SNode *snode) {
    // suffix is for the array size
    auto node_name = snode->node_type_name;
    auto struct_name = snode->get_node_type_name_hinted();

    if (snode->type == SNodeType::place) {
      const auto type = opencl_data_type_name(snode->dt);
      emit("{} {};", type, node_name);
      return data_type_size(snode->dt);

    } else if (snode->type == SNodeType::root) {
      emit("struct Ti_{} {{", struct_name);
      size_t size = generate_children(snode);
      emit("}};");
      return size;

    } else if (snode->type == SNodeType::dense) {
      emit("struct Ti_{} {{", struct_name);
      size_t size = generate_children(snode);
      emit("}} {}[{}];", node_name, snode->n);
      return size * snode->n;

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
  auto [source, size] = gen.compile();
  layout_source = source;
  layout_size = size;
  allocate_root_buffer();
}

}  // namespace opencl
TLANG_NAMESPACE_END
