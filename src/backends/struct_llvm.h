#include "struct.h"
#include "llvm_jit.h"

TLANG_NAMESPACE_BEGIN

class StructCompilerLLVM : public StructCompiler {
 public:
  StructCompilerLLVM();

  TaichiLLVMContext *tlctx;
  llvm::LLVMContext *llvm_ctx;
  std::unique_ptr<llvm::Module> module;

  std::map<SNode *, llvm::Function *> chain_accessors;
  std::map<SNode *, llvm::Function *> leaf_accessors;
  std::map<SNode *, std::string> leaf_accessor_names;

  virtual void codegen(SNode &snode) override;

  virtual void generate_leaf_accessors(SNode &snode) override;

  virtual void load_accessors(SNode &snode) override;

  virtual void set_parents(SNode &snode) override;

  virtual void run(SNode &node) override;
};

TLANG_NAMESPACE_END
