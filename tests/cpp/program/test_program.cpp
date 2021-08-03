#include "tests/cpp/program/test_program.h"

namespace taichi {
namespace lang {

void TestProgram::setup() {
  prog_ = std::make_unique<Program>(Arch::x64);
  prog_->materialize_runtime();
  prog_->add_snode_tree(std::make_unique<SNode>(/*depth=*/0, SNodeType::root));
}

}  // namespace lang
}  // namespace taichi
