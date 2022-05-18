#pragma once

#ifdef TI_WITH_LLVM
// Codegen for the hierarchical data structure (LLVM)
#include "taichi/llvm/llvm_program.h"
#include "taichi/llvm/llvm_codegen_utils.h"
#include "taichi/struct/struct.h"

namespace taichi {
namespace lang {

class LlvmProgramImpl;
class StructCompilerLLVM : public StructCompiler, public LLVMModuleBuilder {
 public:
  StructCompilerLLVM(Arch arch,
                     const CompileConfig *config,
                     TaichiLLVMContext *tlctx,
                     std::unique_ptr<llvm::Module> &&module,
                     int snode_tree_id);

  StructCompilerLLVM(Arch arch,
                     LlvmProgramImpl *prog,
                     std::unique_ptr<llvm::Module> &&module,
                     int snode_tree_id);

  void generate_types(SNode &snode) override;

  void generate_child_accessors(SNode &snode) override;

  void run(SNode &node) override;

  llvm::Function *create_function(llvm::FunctionType *ft,
                                  std::string func_name);

  void generate_refine_coordinates(SNode *snode);

  static std::string type_stub_name(SNode *snode);

  static llvm::Type *get_stub(llvm::Module *module, SNode *snode, uint32 index);

  static llvm::Type *get_llvm_node_type(llvm::Module *module, SNode *snode);

  static llvm::Type *get_llvm_body_type(llvm::Module *module, SNode *snode);

  static llvm::Type *get_llvm_aux_type(llvm::Module *module, SNode *snode);

  static llvm::Type *get_llvm_element_type(llvm::Module *module, SNode *snode);

 private:
  Arch arch_;
  const CompileConfig *const config_;
  TaichiLLVMContext *const tlctx_;
  llvm::LLVMContext *const llvm_ctx_;
  int snode_tree_id_;
};

}  // namespace lang
}  // namespace taichi

#endif  //#ifdef TI_WITH_LLVM
