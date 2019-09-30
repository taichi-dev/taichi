// Codegen for the hierarchical data structure
#pragma once

#include "../snode.h"
#include "base.h"
#include "loopgen.h"

TLANG_NAMESPACE_BEGIN

class StructCompiler : public CodeGenBase {
 public:
  std::vector<SNode *> stack;
  std::vector<SNode *> snodes;
  std::vector<SNode *> ambient_snodes;
  std::string root_type;
  std::function<void *()> creator;
  std::function<void()> profiler_print;
  std::function<void()> profiler_clear;
  LoopGenerator loopgen;

  StructCompiler();

  void collect_snodes(SNode &snode);

  // propagate root-to-leaf for a well-formed data structure
  void infer_snode_properties(SNode &snode);

  // generate C++/llvm IR
  virtual void generate_types(SNode &snode);

  virtual void generate_leaf_accessors(SNode &snode);

  virtual void load_accessors(SNode &snode);

  virtual void run(SNode &node, bool host);

  static std::unique_ptr<StructCompiler> make(bool use_llvm, Arch arch);
};

TLANG_NAMESPACE_END
