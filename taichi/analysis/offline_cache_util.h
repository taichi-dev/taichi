#pragma once

#include <string>

#include "taichi/rhi/arch.h"

namespace taichi::lang {

struct CompileConfig;
class Program;
class IRNode;
class SNode;
class Kernel;

std::string get_hashed_offline_cache_key_of_snode(SNode *snode);
std::string get_hashed_offline_cache_key(const CompileConfig &config,
                                         Kernel *kernel);
void gen_offline_cache_key(Program *prog, IRNode *ast, std::ostream *os);

}  // namespace taichi::lang
