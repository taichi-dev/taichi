#pragma once

#include <vector>

#include "taichi/ir/stmt_op_types.h"

namespace taichi {
namespace lang {

class LinearizeStmt;
class SNode;
class Stmt;
class StructForStmt;
class VecStatement;

/**
 * Lowers an SNode at a given indices to a series of concrete ops.
 */
class ScalarPointerLowerer {
 public:
  /**
   * Constructor
   *
   * @param leaf_snode: SNode of the accessed field
   * @param indices: Indices to access the field
   * @param snode_op: SNode operation
   * @param is_bit_vectorized: Is @param leaf_snode bit vectorized
   * @param lowered: Collects the output ops
   */
  explicit ScalarPointerLowerer(SNode *leaf_snode,
                                const std::vector<Stmt *> &indices,
                                const SNodeOpType snode_op,
                                const bool is_bit_vectorized,
                                VecStatement *lowered,
                                const bool packed);

  virtual ~ScalarPointerLowerer() = default;
  /**
   * Runs the lowering process.
   *
   * This can only be called once.
   */
  void run();

 protected:
  /**
   * Handles the SNode at a given @param level.
   *
   * @param level: Level of the SNode in the access path
   * @param linearized: Linearized indices statement for this level
   * @param last: SNode access op (e.g. GetCh) of the last iteration
   */
  virtual Stmt *handle_snode_at_level(int level,
                                      LinearizeStmt *linearized,
                                      Stmt *last) {
    return last;
  }

  std::vector<SNode *> snodes() const {
    return snodes_;
  }

  int path_length() const {
    return path_length_;
  }

  const std::vector<Stmt *> indices_;
  const SNodeOpType snode_op_;
  const bool is_bit_vectorized_;
  VecStatement *const lowered_;
  const bool packed_;

 private:
  std::vector<SNode *> snodes_;
  int path_length_{0};
};

}  // namespace lang
}  // namespace taichi
