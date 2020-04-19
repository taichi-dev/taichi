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

  static llvm::Type *get_stub(llvm::Module *module,
                              SNode *snode,
                              uint32 index) {
    TI_ASSERT(module);
    TI_ASSERT(snode);
    auto stub = module->getTypeByName(type_stub_name(snode));
    TI_ASSERT(stub);
    TI_ASSERT(stub->getStructNumElements() == 4);
    TI_ASSERT(0 <= index && index < 4);
    auto type = stub->getContainedType(index);
    TI_ASSERT(type);
    return type;
  }

  static llvm::Type *get_llvm_node_type(llvm::Module *module, SNode *snode) {
    return get_stub(module, snode, 0);
  }

  static llvm::Type *get_llvm_body_type(llvm::Module *module, SNode *snode) {
    return get_stub(module, snode, 1);
  }

  static llvm::Type *get_llvm_aux_type(llvm::Module *module, SNode *snode) {
    return get_stub(module, snode, 2);
  }

  static llvm::Type *get_llvm_element_type(llvm::Module *module, SNode *snode) {
    return get_stub(module, snode, 3);
  }
};

TLANG_NAMESPACE_END
