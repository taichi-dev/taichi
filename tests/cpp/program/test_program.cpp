#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

void TestProgram::setup(Arch arch) {
  prog_ = std::make_unique<Program>(arch);
  prog_->materialize_runtime();
  prog_->add_snode_tree(std::make_unique<SNode>(/*depth=*/0, SNodeType::root),
                        /*compile_only=*/false);
}

}  // namespace lang
}  // namespace taichi
