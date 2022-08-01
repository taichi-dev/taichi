#pragma once

#include "taichi/ir/ir.h"
#include "taichi/ir/mesh.h"
#include "taichi/ir/pass.h"
#include "taichi/analysis/gather_uniquely_accessed_pointers.h"
#include "taichi/analysis/mesh_bls_analyzer.h"
#include <atomic>
#include <optional>
#include <unordered_set>
#include <unordered_map>

namespace taichi {
namespace lang {

class DiffRange {
 private:
  bool related_;

 public:
  int coeff;
  int low, high;

  DiffRange() : DiffRange(false, 0) {
  }

  DiffRange(bool related, int coeff) : DiffRange(related, 0, 0) {
    TI_ASSERT(related == false);
  }

  DiffRange(bool related, int coeff, int low)
      : DiffRange(related, coeff, low, low + 1) {
  }

  DiffRange(bool related, int coeff, int low, int high)
      : related_(related), coeff(coeff), low(low), high(high) {
    if (!related) {
      this->low = this->high = 0;
    }
  }

  bool related() const {
    return related_;
  }

  bool linear_related() const {
    return related_ && coeff == 1;
  }

  bool certain() {
    TI_ASSERT(related_);
    return high == low + 1;
  }
};

enum AliasResult { same, uncertain, different };

class ControlFlowGraph;

// IR Analysis
namespace irpass {
namespace analysis {

/**
 * Checks if the two input statements may be aliased to the same address.
 *
 * @param val1
 *   The first statement to check.
 *
 * @param val2
 *   The second statement to check.
 *
 * @return
 *   The analyzed result.
 */
AliasResult alias_analysis(Stmt *var1, Stmt *var2);

std::unique_ptr<ControlFlowGraph> build_cfg(IRNode *root);
void check_fields_registered(IRNode *root);
std::unique_ptr<IRNode> clone(IRNode *root, Kernel *kernel = nullptr);
int count_statements(IRNode *root);

/**
 * Checks if the two input statements definitely point to the same address.
 *
 * @param val1
 *   The first statement to check.
 *
 * @param val2
 *   The second statement to check.
 *
 * @return
 *   Returns true iff. the two stmts definitely point to the same address.
 */
bool definitely_same_address(Stmt *var1, Stmt *var2);

std::unordered_set<Stmt *> detect_fors_with_break(IRNode *root);
std::unordered_set<Stmt *> detect_loops_with_continue(IRNode *root);
std::unordered_set<SNode *> gather_deactivations(IRNode *root);
std::pair<std::unordered_set<SNode *>, std::unordered_set<SNode *>>
gather_snode_read_writes(IRNode *root);
std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test);
void gather_uniquely_accessed_bit_structs(IRNode *root, AnalysisManager *amgr);
std::pair<std::unordered_map<const SNode *, GlobalPtrStmt *>,
          std::unordered_map<int, ExternalPtrStmt *>>
gather_uniquely_accessed_pointers(IRNode *root);
std::unique_ptr<std::unordered_set<AtomicOpStmt *>> gather_used_atomics(
    IRNode *root);
std::vector<Stmt *> get_load_pointers(Stmt *load_stmt);

Stmt *get_store_data(Stmt *store_stmt);
std::vector<Stmt *> get_store_destination(Stmt *store_stmt);
bool has_store_or_atomic(IRNode *root, const std::vector<Stmt *> &vars);
std::pair<bool, Stmt *> last_store_or_atomic(IRNode *root, Stmt *var);

/**
 * Checks if the two input statements may point to the same address.
 *
 * @param val1
 *   The first statement to check.
 *
 * @param val2
 *   The second statement to check.
 *
 * @return
 *   Returns false iff. the two stmts are definitely not aliased.
 */
bool maybe_same_address(Stmt *var1, Stmt *var2);

/**
 * Test if root1 and root2 are the same, i.e., have the same type,
 * the same operands, the same fields, and the same containing statements.
 *
 * @param root1
 *   The first root to check.
 *
 * @param root2
 *   The second root to check.
 *
 * @param id_map
 *   If id_map is std::nullopt by default, two operands are considered
 *   the same if they have the same id and do not belong to either root,
 *   or they belong to root1 and root2 at the same position in the roots.
 *   Otherwise, this function also recursively check the operands until
 *   ids in the id_map are reached.
 */
bool same_statements(
    IRNode *root1,
    IRNode *root2,
    const std::optional<std::unordered_map<int, int>> &id_map = std::nullopt);

/**
 * Test if stmt1 and stmt2 definitely have the same value.
 * Any global fields can be modified between stmt1 and stmt2.
 *
 * @param id_map
 *   Same as in same_statements(root1, root2, id_map).
 */
bool same_value(
    Stmt *stmt1,
    Stmt *stmt2,
    const std::optional<std::unordered_map<int, int>> &id_map = std::nullopt);

DiffRange value_diff_loop_index(Stmt *stmt, Stmt *loop, int index_id);

/**
 * Result of the value_diff_ptr_index pass.
 */
struct DiffPtrResult {
  // Whether the difference of the checked statements is certain.
  bool is_diff_certain{false};
  // The difference of the value of two statements (i.e. val1 - val2). This is
  // meaningful only when |is_diff_certain| is true.
  int diff_range{0};

  static DiffPtrResult make_certain(int diff) {
    return DiffPtrResult{/*is_diff_certain=*/true, /*diff_range=*/diff};
  }
  static DiffPtrResult make_uncertain() {
    return DiffPtrResult{/*is_diff_certain=*/false, /*diff_range=*/0};
  }
};

/**
 * Checks if the difference of the value of the two statements is certain.
 *
 * @param val1
 *   The first statement to check.
 *
 * @param val2
 *   The second statement to check.
 */
DiffPtrResult value_diff_ptr_index(Stmt *val1, Stmt *val2);

std::unordered_set<Stmt *> constexpr_prop(
    Block *block,
    std::function<bool(Stmt *)> is_const_seed);

void verify(IRNode *root);

// Mesh Related.
void gather_meshfor_relation_types(IRNode *node);
std::pair</* owned= */ std::unordered_set<mesh::MeshElementType>,
          /* total= */ std::unordered_set<mesh::MeshElementType>>
gather_mesh_thread_local(OffloadedStmt *offload, const CompileConfig &config);
std::unique_ptr<MeshBLSCaches> initialize_mesh_local_attribute(
    OffloadedStmt *offload,
    bool auto_mesh_local,
    const CompileConfig &config);

}  // namespace analysis
}  // namespace irpass
}  // namespace lang
}  // namespace taichi
