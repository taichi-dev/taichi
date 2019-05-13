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
  LoopGenerator loopgen;

  StructCompiler();

  std::string create_snode() {
    TC_ASSERT(snode_count < 10000);
    return fmt::format("S{}", snode_count++);
  }

  void visit(SNode &snode);

  void generate_leaf_accessors(SNode &snode);

  void load_accessors(SNode &snode);

  void set_parents(SNode &snode);

  void run(SNode &node);
};

TLANG_NAMESPACE_END
