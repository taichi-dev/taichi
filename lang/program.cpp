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
  layout_fn = scomp.get_source_fn();
}

// All data structure originates from a "root", which is a forked node.

auto test_snode = [&]() {
  Program prog(Arch::x86_64);

  auto i = Expr::index(0);
  auto u = variable(DataType::i32);

  int n = 128;

  prog.layout([&] { root.fixed(i, n).place(u); });

  for (int i = 0; i < n; i++) {
    u.set<int32>(i, i + 1);
  }

  for (int i = 0; i < n; i++) {
    TC_ASSERT(u.get<int32>(i) == i + 1);
  }
};

TC_REGISTER_TASK(test_snode);

TLANG_NAMESPACE_END
