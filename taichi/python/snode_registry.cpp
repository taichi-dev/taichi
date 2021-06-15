#include "taichi/python/snode_registry.h"

#include "taichi/ir/snode.h"

namespace taichi {
namespace lang {

SNode *SNodeRegistry::create_root() {
  auto n = std::make_unique<SNode>(/*depth=*/0, SNodeType::root);
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
