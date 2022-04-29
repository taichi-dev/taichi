#pragma once

#include <string>

namespace taichi {
namespace lang {

struct CompileConfig;
class Program;
class IRNode;
class SNode;
class Kernel;

std::string get_hashed_offline_cache_key_of_snode(SNode *snode);
std::string get_hashed_offline_cache_key(CompileConfig *config, Kernel *kernel);
void gen_offline_cache_key(Program *prog, IRNode *ast, std::ostream *os);

}  // namespace lang
}  // namespace taichi
