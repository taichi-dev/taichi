// Codegen for the hierarchical data structure (LLVM)

#include "taichi/struct/struct.h"
#include "taichi/llvm/llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

class StructCompilerLLVM : public StructCompiler, public LLVMModuleBuilder {
 public:
  StructCompilerLLVM(Program *prog, Arch arch);

  Arch arch;
  TaichiLLVMContext *tlctx;
  llvm::LLVMContext *llvm_ctx;

  void generate_types(SNode &snode) override;

  void generate_child_accessors(SNode &snode) override;

  void run(SNode &node, bool host) override;

  void generate_refine_coordinates(SNode *snode);

  static std::string type_stub_name(SNode *snode);

  static llvm::Type *get_llvm_body_type(llvm::Module *module,
                                        SNode *snode) {
    auto stub = module->getTypeByName(type_stub_name(snode));
    return stub->getContainedType(2);
  }
};

TLANG_NAMESPACE_END
