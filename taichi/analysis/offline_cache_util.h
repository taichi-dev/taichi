#pragma once

#include <string>

namespace taichi {
namespace lang {

struct CompileConfig;
class Kernel;
class SNode;

std::string get_hashed_offline_cache_key_of_snode(SNode *snode);
std::string get_hashed_offline_cache_key(CompileConfig *config, Kernel *kernel);

}  // namespace lang
}  // namespace taichi
