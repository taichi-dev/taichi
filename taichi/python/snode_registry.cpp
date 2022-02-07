#include "taichi/python/snode_registry.h"

#include "taichi/common/logging.h"
#include "taichi/ir/snode.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

SNode *SNodeRegistry::create_root(Program *prog) {
  TI_ASSERT(prog != nullptr);
  auto n = std::make_unique<SNode>(/*depth=*/0, SNodeType::root,
                                   prog->get_snode_to_glb_var_exprs(),
                                   &prog->get_snode_rw_accessors_bank());
  auto *res = n.get();
  snodes_.push_back(std::move(n));
  return res;
}

std::unique_ptr<SNode> SNodeRegistry::finalize(const SNode *snode) {
  for (auto it = snodes_.begin(); it != snodes_.end(); ++it) {
    if (it->get() == snode) {
      auto res = std::move(*it);
      snodes_.erase(it);
      return res;
    }
  }
  return nullptr;
}

}  // namespace lang
}  // namespace taichi
