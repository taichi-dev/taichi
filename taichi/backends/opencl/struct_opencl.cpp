#include "opencl_program.h"

#include "taichi/util/line_appender.h"
#include "taichi/ir/snode.h"

TLANG_NAMESPACE_BEGIN
namespace opencl {

namespace {

// Generate corresponding OpenCL Source Code for Taichi Structures
class OpenclLayoutGen {
 public:
  OpenclLayoutGen(OpenclProgram *program, SNode *root)
      : program(program), root(root) {
  }

  std::string compile() {
    return "233";
  }

 private:
  void generate_children(SNode *snode);
  void generate_types(SNode *snode);

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    line_appender.append(std::move(f), std::move(args)...);
  }

  OpenclProgram *program;

  SNode *root;
  std::vector<SNode *> snodes;
  LineAppender line_appender;
};

}  // namespace

void OpenclProgram::compile_layout(SNode *root) {
  OpenclLayoutGen gen(this, root);
  layout_source = gen.compile();
}

}  // namespace opencl
TLANG_NAMESPACE_END
