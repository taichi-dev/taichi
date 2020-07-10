#pragma once

#include "taichi/ir/ir.h"
#include <atomic>
#include <unordered_set>
#include <unordered_map>

TLANG_NAMESPACE_BEGIN

class DiffRange {
 private:
  bool related;

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
      : related(related), coeff(coeff), low(low), high(high) {
    if (!related) {
      this->low = this->high = 0;
    }
  }

  bool related_() const {
    return related;
  }

  bool linear_related() const {
    return related && coeff == 1;
  }

  bool certain() {
    TI_ASSERT(related);
    return high == low + 1;
  }
};

enum AliasResult { same, uncertain, different };

class ControlFlowGraph;

// IR Analysis
namespace irpass::analysis {

AliasResult alias_analysis(Stmt *var1, Stmt *var2);
std::unique_ptr<ControlFlowGraph> build_cfg(IRNode *root);
void check_fields_registered(IRNode *root);
std::unique_ptr<IRNode> clone(IRNode *root, Kernel *kernel = nullptr);
int count_statements(IRNode *root);
bool definitely_same_address(Stmt *var1, Stmt *var2);
std::unordered_set<Stmt *> detect_fors_with_break(IRNode *root);
std::unordered_set<Stmt *> detect_loops_with_continue(IRNode *root);
std::unordered_set<SNode *> gather_deactivations(IRNode *root);
std::vector<Stmt *> gather_statements(IRNode *root,
                                      const std::function<bool(Stmt *)> &test);
std::unique_ptr<std::unordered_set<AtomicOpStmt *>> gather_used_atomics(
    IRNode *root);
std::vector<Stmt *> get_load_pointers(Stmt *load_stmt);
Stmt *get_store_data(Stmt *store_stmt);
std::vector<Stmt *> get_store_destination(Stmt *store_stmt);
bool has_store_or_atomic(IRNode *root, const std::vector<Stmt *> &vars);
std::pair<bool, Stmt *> last_store_or_atomic(IRNode *root, Stmt *var);
bool maybe_same_address(Stmt *var1, Stmt *var2);
bool same_statements(IRNode *root1, IRNode *root2);
DiffRange value_diff(Stmt *stmt, int lane, Stmt *alloca);
DiffRange value_diff_loop_index(Stmt *stmt, Stmt *loop, int index_id);
void verify(IRNode *root);

}  // namespace irpass::analysis

TLANG_NAMESPACE_END
