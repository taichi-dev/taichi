// Codegen for the hierarchical data structure (LLVM)

#include "struct.h"
#include "llvm_jit.h"
#include "llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

class StructCompilerLLVM : public StructCompiler, public ModuleBuilder {
 public:
  StructCompilerLLVM(Arch arch);

  Arch arch;
  TaichiLLVMContext *tlctx;
  llvm::LLVMContext *llvm_ctx;

  std::map<SNode *, llvm::Function *> chain_accessors;
  std::map<SNode *, llvm::Function *> leaf_accessors;
  std::map<SNode *, std::string> leaf_accessor_names;

  virtual void generate_types(SNode &snode) override;

  virtual void generate_leaf_accessors(SNode &snode) override;

  virtual void load_accessors(SNode &snode) override;

  virtual void run(SNode &node) override;

  void generate_refine_coordinates(SNode *snode);
};

TLANG_NAMESPACE_END
