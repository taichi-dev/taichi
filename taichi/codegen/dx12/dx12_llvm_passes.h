
#pragma once

#include <string>
#include <vector>

namespace llvm {
class Function;
class Module;
}  // namespace llvm

namespace taichi {
namespace lang {
struct CompileConfig;

namespace directx12 {

void mark_function_as_cs_entry(llvm::Function *);
bool is_cs_entry(llvm::Function *);
void set_num_threads(llvm::Function *, unsigned x, unsigned y, unsigned z);

std::vector<uint8_t> global_optimize_module(llvm::Module *module,
                                            CompileConfig &config);

extern const char *NumWorkGroupsCBName;

}  // namespace directx12
}  // namespace lang
}  // namespace taichi

