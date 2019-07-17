#include "struct.h"

TLANG_NAMESPACE_BEGIN

class StructCompilerLLVM : public StructCompiler {
 public:
  StructCompilerLLVM();

  virtual void visit(SNode &snode) override;

  virtual void generate_leaf_accessors(SNode &snode) override;

  virtual void load_accessors(SNode &snode) override;

  virtual void set_parents(SNode &snode) override;

  virtual void run(SNode &node) override;
};

TLANG_NAMESPACE_END
