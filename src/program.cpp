#include <taichi/common/task.h>
#include "program.h"
#include "structural_node.h"
#include "codegen/structt.h"

TLANG_NAMESPACE_BEGIN

Program *current_program = nullptr;
SNode root;

void Program::materialize_layout() {
  StructCompiler scomp;
  scomp.run(root);
  layout_fn = scomp.get_source_fn();
  data_structure = scomp.creator();
}

TLANG_NAMESPACE_END
