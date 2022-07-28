#pragma once

#include <atomic>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "taichi/ir/control_flow_graph.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/pass.h"
#include "taichi/transforms/check_out_of_bound.h"
#include "taichi/transforms/constant_fold.h"
#include "taichi/transforms/inlining.h"
#include "taichi/transforms/lower_access.h"
#include "taichi/transforms/make_block_local.h"
#include "taichi/transforms/make_mesh_block_local.h"
#include "taichi/transforms/demote_mesh_statements.h"
#include "taichi/transforms/simplify.h"
#include "taichi/common/trait.h"

TLANG_NAMESPACE_BEGIN

class ScratchPads;

class Function;

// IR passes
namespace irpass {

void re_id(IRNode *root);
void flag_access(IRNode *root);
bool die(IRNode *root);
bool simplify(IRNode *root, const CompileConfig &config);
bool cfg_optimization(
    IRNode *root,
    bool after_lower_access,
    bool autodiff_enabled,
    const std::optional<ControlFlowGraph::LiveVarAnalysisConfig>
        &lva_config_opt = std::nullopt);
bool alg_simp(IRNode *root, const CompileConfig &config);
bool demote_operations(IRNode *root, const CompileConfig &config);
bool binary_op_simplify(IRNode *root, const CompileConfig &config);
bool whole_kernel_cse(IRNode *root);
bool extract_constant(IRNode *root, const CompileConfig &config);
bool unreachable_code_elimination(IRNode *root);
bool loop_invariant_code_motion(IRNode *root, const CompileConfig &config);
void full_simplify(IRNode *root,
                   const CompileConfig &config,
                   const FullSimplifyPass::Args &args);
void print(IRNode *root, std::string *output = nullptr);
void frontend_type_check(IRNode *root);
void lower_ast(IRNode *root);
void type_check(IRNode *root, const CompileConfig &config);
bool inlining(IRNode *root,
              const CompileConfig &config,
              const InliningPass::Args &args);
void bit_loop_vectorize(IRNode *root);
void slp_vectorize(IRNode *root);
void replace_all_usages_with(IRNode *root, Stmt *old_stmt, Stmt *new_stmt);
bool check_out_of_bound(IRNode *root,
                        const CompileConfig &config,
                        const CheckOutOfBoundPass::Args &args);
void make_thread_local(IRNode *root, const CompileConfig &config);
std::unique_ptr<ScratchPads> initialize_scratch_pad(OffloadedStmt *root);
void make_block_local(IRNode *root,
                      const CompileConfig &config,
                      const MakeBlockLocalPass::Args &args);
void make_mesh_thread_local(IRNode *root,
                            const CompileConfig &config,
                            const MakeBlockLocalPass::Args &args);
void make_mesh_block_local(IRNode *root,
                           const CompileConfig &config,
                           const MakeMeshBlockLocal::Args &args);
void demote_mesh_statements(IRNode *root,
                            const CompileConfig &config,
                            const DemoteMeshStatements::Args &args);
bool remove_loop_unique(IRNode *root);
bool remove_range_assumption(IRNode *root);
bool lower_access(IRNode *root,
                  const CompileConfig &config,
                  const LowerAccessPass::Args &args);
void auto_diff(IRNode *root,
               const CompileConfig &config,
               AutodiffMode autodiffMode,
               bool use_stack = false);
/**
 * Determine all adaptive AD-stacks' size. This pass is idempotent, i.e.,
 * there are no side effects if called more than once or called when not needed.
 * @return Whether the IR is modified, i.e., whether there exists adaptive
 * AD-stacks before this pass.
 */
bool determine_ad_stack_size(IRNode *root, const CompileConfig &config);
bool constant_fold(IRNode *root,
                   const CompileConfig &config,
                   const ConstantFoldPass::Args &args);
void offload(IRNode *root, const CompileConfig &config);
bool transform_statements(
    IRNode *root,
    std::function<bool(Stmt *)> filter,
    std::function<void(Stmt *, DelayedIRModifier *)> transformer);
/**
 * @param root The IR root to be traversed.
 * @param filter A function which tells if a statement need to be replaced.
 * @param generator If a statement |s| need to be replaced, generate a new
 * statement |s1| with the argument |s|, insert |s1| to where |s| is defined,
 * remove |s|'s definition, and replace all usages of |s| with |s1|.
 * @return Whether the IR is modified.
 */
bool replace_and_insert_statements(
    IRNode *root,
    std::function<bool(Stmt *)> filter,
    std::function<std::unique_ptr<Stmt>(Stmt *)> generator);
/**
 * @param finder If a statement |s| need to be replaced, find the existing
 * statement |s1| with the argument |s|, remove |s|'s definition, and replace
 * all usages of |s| with |s1|.
 */
bool replace_statements(IRNode *root,
                        std::function<bool(Stmt *)> filter,
                        std::function<Stmt *(Stmt *)> finder);
void demote_dense_struct_fors(IRNode *root, bool packed);
void demote_no_access_mesh_fors(IRNode *root);
bool demote_atomics(IRNode *root, const CompileConfig &config);
void reverse_segments(IRNode *root);  // for autograd
void detect_read_only(IRNode *root);
void optimize_bit_struct_stores(IRNode *root,
                                const CompileConfig &config,
                                AnalysisManager *amgr);

ENUM_FLAGS(ExternalPtrAccess){NONE = 0, READ = 1, WRITE = 2};

/**
 * Checks the access to external pointers in an offload.
 *
 * @param val1
 *   The offloaded statement to check
 *
 * @return
 *   The analyzed result.
 */
std::unordered_map<int, ExternalPtrAccess> detect_external_ptr_access_in_task(
    OffloadedStmt *offload);

// compile_to_offloads does the basic compilation to create all the offloaded
// tasks of a Taichi kernel. It's worth pointing out that this doesn't demote
// dense struct fors. This is a necessary workaround to prevent the async
// engine from fusing incompatible offloaded tasks. TODO(Lin): check this
// comment
void compile_to_offloads(IRNode *ir,
                         const CompileConfig &config,
                         Kernel *kernel,
                         bool verbose,
                         AutodiffMode autodiff_mode,
                         bool ad_use_stack,
                         bool start_from_ast);

void offload_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           Kernel *kernel,
                           bool verbose,
                           bool determine_ad_stack_size,
                           bool lower_global_access,
                           bool make_thread_local,
                           bool make_block_local);
// compile_to_executable fully covers compile_to_offloads, and also does
// additional optimizations so that |ir| can be directly fed into codegen.
void compile_to_executable(IRNode *ir,
                           const CompileConfig &config,
                           Kernel *kernel,
                           AutodiffMode autodiff_mode,
                           bool ad_use_stack,
                           bool verbose,
                           bool lower_global_access = true,
                           bool make_thread_local = false,
                           bool make_block_local = false,
                           bool start_from_ast = true);
// Compile a function with some basic optimizations, so that the number of
// statements is reduced before inlining.
void compile_function(IRNode *ir,
                      const CompileConfig &config,
                      Function *func,
                      AutodiffMode autodiff_mode,
                      bool verbose,
                      bool start_from_ast);
}  // namespace irpass

TLANG_NAMESPACE_END
