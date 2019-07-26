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
  int snode_count;
  void *(*creator)();
  void (*profiler_print)();
  void (*profiler_clear)();
  LoopGenerator loopgen;

  StructCompiler();

  std::string create_snode() {
    TC_ASSERT(snode_count < 10000);
    return fmt::format("S{}", snode_count++);
  }

  // propagate root-to-leaf for a well-formed data structure
  virtual void compile(SNode &snode);

  // generate C++/llvm IR
  virtual void codegen(SNode &snode);

  virtual void generate_leaf_accessors(SNode &snode);

  virtual void load_accessors(SNode &snode);

  virtual void set_parents(SNode &snode);

  virtual void run(SNode &node);

  static std::unique_ptr<StructCompiler> make(bool use_llvm);
};

TLANG_NAMESPACE_END
