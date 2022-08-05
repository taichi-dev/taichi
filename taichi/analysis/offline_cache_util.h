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

namespace offline_cache {

std::string mangle_name(const std::string &primal_name, const std::string &key);
bool try_demangle_name(const std::string &mangled_name, std::string &primal_name, std::string &key);

}  // namespace offline_cache

}  // namespace lang
}  // namespace taichi
