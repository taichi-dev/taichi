// Codegen for the hierarchical data structure
#pragma once

#include "taichi/ir/snode.h"
#include "taichi/program.h"

TLANG_NAMESPACE_BEGIN

class StructCompiler {
 public:
  std::vector<SNode *> stack;
  std::vector<SNode *> snodes;
  std::vector<SNode *> ambient_snodes;
  std::size_t root_size;
  Program *prog;

  SNodeAttributes snode_attr;

  explicit StructCompiler(Program *prog);

  virtual ~StructCompiler() = default;

  void collect_snodes(SNode &snode);

  // propagate root-to-leaf for a well-formed data structure
  void infer_snode_properties(SNode &snode);

  // generate C++/llvm IR
  virtual void generate_types(SNode &snode);

  virtual void generate_leaf_accessors(SNode &snode);

  virtual void run(SNode &node, bool host);

  static std::unique_ptr<StructCompiler> make(Program *prog, Arch arch);
};

TLANG_NAMESPACE_END
