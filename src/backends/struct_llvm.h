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

  llvm::Type *get_runtime_type(const std::string &name) {
    auto ty = module->getTypeByName("struct." + name);
    if (!ty) {
      TC_ERROR("Runtime type {} not found.", name);
    }
    return ty;
  }

  llvm::Function *get_runtime_function(const std::string &name) {
    auto f = module->getFunction(name);
    if (!f) {
      TC_ERROR("Runtime function {} not found.", name);
    }
    return f;
  }

  virtual void codegen(SNode &snode) override;

  virtual void generate_leaf_accessors(SNode &snode) override;

  void emit_element_list_gen(SNode *snode);

  virtual void load_accessors(SNode &snode) override;

  virtual void set_parents(SNode &snode) override;

  virtual void run(SNode &node) override;

  void generate_refine_coordinates(SNode *snode);
};

TLANG_NAMESPACE_END
