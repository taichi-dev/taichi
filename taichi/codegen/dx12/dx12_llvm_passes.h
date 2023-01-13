
#pragma once

#include <string>
#include <vector>

namespace llvm {
class Function;
class Module;
class Type;
class GlobalVariable;
}  // namespace llvm

namespace taichi::lang {
struct CompileConfig;

namespace directx12 {

void mark_function_as_cs_entry(llvm::Function *);
bool is_cs_entry(llvm::Function *);
void set_num_threads(llvm::Function *, unsigned x, unsigned y, unsigned z);
llvm::GlobalVariable *createGlobalVariableForResource(llvm::Module &M,
                                                      const char *Name,
                                                      llvm::Type *Ty);

std::vector<uint8_t> global_optimize_module(llvm::Module *module,
                                            const CompileConfig &config);

extern const char *NumWorkGroupsCBName;

}  // namespace directx12
}  // namespace taichi::lang

namespace llvm {
class ModulePass;
class PassRegistry;
class Function;

/// Initializer for DXIL-prepare
void initializeTaichiRuntimeContextLowerPass(PassRegistry &);

/// Pass to convert modules into DXIL-compatable modules
ModulePass *createTaichiRuntimeContextLowerPass();

/// Initializer for taichi intrinsic lower.
void initializeTaichiIntrinsicLowerPass(PassRegistry &);

/// Pass to lower taichi intrinsic into DXIL intrinsic.
ModulePass *createTaichiIntrinsicLowerPass(
    const taichi::lang::CompileConfig *config);

}  // namespace llvm
