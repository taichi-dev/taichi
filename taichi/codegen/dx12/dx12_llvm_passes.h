
#pragma once

#include <string>
#include <vector>

namespace llvm {
class Function;
class Module;
class Type;
class GlobalVariable;
}  // namespace llvm

namespace taichi {
namespace lang {
struct CompileConfig;

namespace directx12 {

// Different Buf in different space.
// Buffer ID keep the same in DXIL.
enum class BufferSpaceId {
  Arr = 0,
  Root = 1,
  Gtmp = 2,
  Args = 3,
  Runtime = 4,
  Result = 5,
  UtilCBuffer = 6,  // For things like Num work groups.
};

enum ResourceAddressSpace {
  CBuffer = 4,
};

void mark_function_as_cs_entry(llvm::Function *);
bool is_cs_entry(llvm::Function *);
void set_num_threads(llvm::Function *, unsigned x, unsigned y, unsigned z);

std::vector<uint8_t> global_optimize_module(llvm::Module *module,
                                            CompileConfig &config);

llvm::GlobalVariable *createGlobalVariableForResource(llvm::Module &M,
                                                      const char *Name,
                                                      llvm::Type *Ty,
                                                      unsigned AddressSpace);

extern const char *NumWorkGroupsCBName;

}  // namespace directx12
}  // namespace lang
}  // namespace taichi

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
ModulePass *createTaichiIntrinsicLowerPass(taichi::lang::CompileConfig *config);

}  // namespace llvm
