// Codegen for the hierarchical data structure (LLVM)

#include "struct.h"
#include "llvm_jit.h"
#include "llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

class StructCompilerLLVM : public StructCompiler, public ModuleBuilder {
 public:
  StructCompilerLLVM(Arch arch);

  SNodeAttributes snode_attr;

  Arch arch;
  TaichiLLVMContext *tlctx;
  llvm::LLVMContext *llvm_ctx;

  void generate_types(SNode &snode) override;

  void generate_leaf_accessors(SNode &snode) override;

  void load_accessors(SNode &snode) override;

  void run(SNode &node, bool host) override;

  void generate_refine_coordinates(SNode *snode);
};

TLANG_NAMESPACE_END
