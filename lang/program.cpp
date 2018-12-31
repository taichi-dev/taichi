#include <taichi/common/task.h>
#include "program.h"
#include "structural_node.h"
#include "struct_compiler.h"

TLANG_NAMESPACE_BEGIN

Program *current_program = nullptr;
SNode root;

void Program::materialize_layout() {
  StructCompiler scomp;
  scomp.run(root);
}

// All data structure originates from a "root", which is a forked node.

auto test_snode = [&]() {
  Program prog(Arch::x86_64);

  auto i = Expr::index(0);
  auto u = placeholder(DataType::i32);

  prog.layout([&] { root.fixed(i, 128).place(u); });

  for (int i = 0; i < 10; i++) {
    // u.access(i);
  }
};

TC_REGISTER_TASK(test_snode);

TLANG_NAMESPACE_END
