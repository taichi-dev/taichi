#pragma once

#include <string>

#include "taichi/rhi/arch.h"

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

std::string get_cache_path_by_arch(const std::string &base_path, Arch arch);
bool enabled_wip_offline_cache(bool enable_hint);
std::string mangle_name(const std::string &primal_name, const std::string &key);
bool try_demangle_name(const std::string &mangled_name,
                       std::string &primal_name,
                       std::string &key);

}  // namespace offline_cache

}  // namespace lang
}  // namespace taichi
