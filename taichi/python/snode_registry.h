#pragma once

#include <vector>
#include <memory>

namespace taichi {
namespace lang {

class SNode;
class Program;

/**
 * A helper class to keep the root SNodes that aren't materialized yet.
 *
 * This is an awkward workaround given that pybind11 does not  allow returning a
 * std::unique_ptr instance and then moving it back to C++.
 *
 * Not thread safe.
 */
class SNodeRegistry {
 public:
  /**
   * Create a new root SNode.
   *
   * Note that this registry takes the ownership of the created SNode.
   *
   * @return Pointer to the created SNode.
   */
  SNode *create_root(Program *prog);

  /**
   * Transfers the ownership of @param snode to the caller.
   *
   * @param snode Returned from create_root()
   * @return The transferred root SNode.
   */
  std::unique_ptr<SNode> finalize(const SNode *snode);

 private:
  std::vector<std::unique_ptr<SNode>> snodes_;
};

}  // namespace lang
}  // namespace taichi
