#pragma once

#include "taichi/ir/ir.h"
#include <atomic>
#include <unordered_set>
#include <unordered_map>

TLANG_NAMESPACE_BEGIN

// IR passes
namespace irpass {

struct OffloadedResult {
  // Total size in bytes of the global temporary variables
  std::size_t total_size;
  // Offloaded local variables to its offset in the global tmps memory.
  std::unordered_map<const Stmt *, std::size_t> local_to_global_offset;
};

void re_id(IRNode *root);
void flag_access(IRNode *root);
void die(IRNode *root);
void simplify(IRNode *root, Kernel *kernel = nullptr);
void alg_simp(IRNode *root, const CompileConfig &config);
void whole_kernel_cse(IRNode *root);
void variable_optimization(IRNode *root, bool after_lower_access);
void extract_constant(IRNode *root);
void full_simplify(IRNode *root,
                   const CompileConfig &config,
                   Kernel *kernel = nullptr);
void print(IRNode *root, std::string *output = nullptr);
void lower(IRNode *root);
void convert_into_loop_index(IRNode *root);
void typecheck(IRNode *root, Kernel *kernel = nullptr);
void loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void vector_split(IRNode *root, int max_width, bool serial_schedule);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);
void check_out_of_bound(IRNode *root);
void lower_access(IRNode *root, bool lower_atomic, Kernel *kernel = nullptr);
void make_adjoint(IRNode *root, bool use_stack = false);
void constant_fold(IRNode *root);
OffloadedResult offload(IRNode *root);
void fix_block_parents(IRNode *root);
void replace_statements_with(IRNode *root,
                             std::function<bool(Stmt *)> filter,
                             std::function<std::unique_ptr<Stmt>()> generator);
void demote_dense_struct_fors(IRNode *root);
void demote_atomics(IRNode *root);
void reverse_segments(IRNode *root);  // for autograd
std::unique_ptr<ScratchPads> initialize_scratch_pad(StructForStmt *root);
OffloadedResult compile_to_offloads(IRNode *ir,
                                    const CompileConfig &config,
                                    bool vectorize,
                                    bool grad,
                                    bool ad_use_stack,
                                    bool verbose,
                                    bool lower_global_access = true);

}  // namespace irpass

TLANG_NAMESPACE_END
