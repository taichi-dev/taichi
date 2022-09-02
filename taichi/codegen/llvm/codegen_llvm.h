// The LLVM backend for CPUs/NVPTX/AMDGPU
#pragma once

#include <set>
#include <unordered_map>

#ifdef TI_WITH_LLVM

#include "taichi/ir/ir.h"
#include "taichi/runtime/llvm/launch_arg_info.h"
#include "taichi/codegen/llvm/llvm_codegen_utils.h"
#include "taichi/codegen/llvm/llvm_compiled_data.h"
#include "taichi/program/program.h"

namespace taichi {
namespace lang {

class TaskCodeGenLLVM;

class FunctionCreationGuard {
 public:
  TaskCodeGenLLVM *mb;
  llvm::Function *old_func;
  llvm::Function *body;
  llvm::BasicBlock *old_entry, *allocas, *entry, *old_final, *final;
  llvm::IRBuilder<>::InsertPoint ip;

  FunctionCreationGuard(TaskCodeGenLLVM *mb,
                        std::vector<llvm::Type *> arguments);

  ~FunctionCreationGuard();
};

class TaskCodeGenLLVM : public IRVisitor, public LLVMModuleBuilder {
 public:
  Kernel *kernel;
  IRNode *ir;
  Program *prog;
  std::string kernel_name;
  std::vector<llvm::Value *> kernel_args;
  llvm::Type *context_ty;
  llvm::Type *physical_coordinate_ty;
  llvm::Value *current_coordinates;
  llvm::Value *parent_coordinates{nullptr};
  llvm::Value *block_corner_coordinates{nullptr};
  llvm::GlobalVariable *bls_buffer{nullptr};
  // Mainly for supporting continue stmt
  llvm::BasicBlock *current_loop_reentry;
  // Mainly for supporting break stmt
  llvm::BasicBlock *current_while_after_loop;
  llvm::FunctionType *task_function_type;
  std::unordered_map<Stmt *, llvm::Value *> llvm_val;
  llvm::Function *func;
  OffloadedStmt *current_offload{nullptr};
  std::unique_ptr<OffloadedTask> current_task;
  std::vector<OffloadedTask> offloaded_tasks;
  llvm::BasicBlock *func_body_bb;
  llvm::BasicBlock *final_block;
  std::set<std::string> linked_modules;
  bool returned{false};
  std::unordered_set<int> used_tree_ids;
  std::unordered_set<int> struct_for_tls_sizes;

  std::unordered_map<const Stmt *, std::vector<llvm::Value *>> loop_vars_llvm;

  std::unordered_map<Function *, llvm::Function *> func_map;

  using IRVisitor::visit;
  using LLVMModuleBuilder::call;

  TaskCodeGenLLVM(Kernel *kernel,
                  IRNode *ir = nullptr,
                  std::unique_ptr<llvm::Module> &&module = nullptr);

  Arch current_arch() {
    return kernel->arch;
  }

  void initialize_context();

  llvm::Value *get_arg(int i);

  llvm::Value *get_context();

  llvm::Value *get_tls_base_ptr();

  llvm::Type *get_tls_buffer_type();

  std::vector<llvm::Type *> get_xlogue_argument_types();

  std::vector<llvm::Type *> get_mesh_xlogue_argument_types();

  llvm::Type *get_xlogue_function_type();

  llvm::Type *get_mesh_xlogue_function_type();

  llvm::Value *get_root(int snode_tree_id);

  llvm::Value *get_runtime();

  void emit_struct_meta_base(const std::string &name,
                             llvm::Value *node_meta,
                             SNode *snode);

  std::unique_ptr<RuntimeObject> emit_struct_meta_object(SNode *snode);

  llvm::Value *emit_struct_meta(SNode *snode);

  virtual void emit_to_module();

  void eliminate_unused_functions();

  /**
   * @brief Runs the codegen and produces the compiled result.
   *
   * After this call, `module` and `tasks` will be moved.
   *
   * @return LLVMCompiledData
   */
  virtual LLVMCompiledData run_compilation();
  // For debugging only
  virtual llvm::Value *create_print(std::string tag,
                                    DataType dt,
                                    llvm::Value *value);

  llvm::Value *create_print(std::string tag, llvm::Value *value);

  llvm::Value *cast_pointer(llvm::Value *val,
                            std::string dest_ty_name,
                            int addr_space = 0);

  void emit_list_gen(OffloadedStmt *listgen);

  void emit_gc(OffloadedStmt *stmt);

  llvm::Value *create_call(llvm::Function *func,
                           llvm::ArrayRef<llvm::Value *> args = {});

  llvm::Value *create_call(llvm::Value *func,
                           llvm::FunctionType *func_ty,
                           llvm::ArrayRef<llvm::Value *> args = {});

  llvm::Value *create_call(std::string func_name,
                           llvm::ArrayRef<llvm::Value *> args = {});
  llvm::Value *call(SNode *snode,
                    llvm::Value *node_ptr,
                    const std::string &method,
                    const std::vector<llvm::Value *> &arguments);

  llvm::Function *get_struct_function(const std::string &name, int tree_id);

  template <typename... Args>
  llvm::Value *call_struct_func(int tree_id,
                                const std::string &func_name,
                                Args &&...args);

  void create_increment(llvm::Value *ptr, llvm::Value *value);

  // Direct translation
  void create_naive_range_for(RangeForStmt *for_stmt);

  static std::string get_runtime_snode_name(SNode *snode);

  void visit(Block *stmt_list) override;

  void visit(AllocaStmt *stmt) override;

  void visit(RandStmt *stmt) override;

  virtual void emit_extra_unary(UnaryOpStmt *stmt);

  void visit(DecorationStmt *stmt) override;

  void visit(UnaryOpStmt *stmt) override;

  void visit(BinaryOpStmt *stmt) override;

  void visit(TernaryOpStmt *stmt) override;

  void visit(IfStmt *if_stmt) override;

  void visit(PrintStmt *stmt) override;

  void visit(ConstStmt *stmt) override;

  void visit(WhileControlStmt *stmt) override;

  void visit(ContinueStmt *stmt) override;

  void visit(WhileStmt *stmt) override;

  void visit(RangeForStmt *for_stmt) override;

  void visit(ArgLoadStmt *stmt) override;

  void visit(ReturnStmt *stmt) override;

  void visit(LocalLoadStmt *stmt) override;

  void visit(LocalStoreStmt *stmt) override;

  void visit(AssertStmt *stmt) override;

  void visit(SNodeOpStmt *stmt) override;

  llvm::Value *atomic_add_quant_fixed(llvm::Value *ptr,
                                      llvm::Type *physical_type,
                                      QuantFixedType *qfxt,
                                      llvm::Value *value);

  llvm::Value *atomic_add_quant_int(llvm::Value *ptr,
                                    llvm::Type *physical_type,
                                    QuantIntType *qit,
                                    llvm::Value *value,
                                    bool value_is_signed);

  llvm::Value *to_quant_fixed(llvm::Value *real, QuantFixedType *qfxt);

  virtual llvm::Value *optimized_reduction(AtomicOpStmt *stmt);

  virtual llvm::Value *quant_type_atomic(AtomicOpStmt *stmt);

  virtual llvm::Value *integral_type_atomic(AtomicOpStmt *stmt);

  virtual llvm::Value *atomic_op_using_cas(
      llvm::Value *output_address,
      llvm::Value *val,
      std::function<llvm::Value *(llvm::Value *, llvm::Value *)> op);

  virtual llvm::Value *real_type_atomic(AtomicOpStmt *stmt);

  void visit(AtomicOpStmt *stmt) override;

  void visit(GlobalPtrStmt *stmt) override;

  void visit(PtrOffsetStmt *stmt) override;

  void store_quant_int(llvm::Value *ptr,
                       llvm::Type *physical_type,
                       QuantIntType *qit,
                       llvm::Value *value,
                       bool atomic);

  void store_quant_fixed(llvm::Value *ptr,
                         llvm::Type *physical_type,
                         QuantFixedType *qfxt,
                         llvm::Value *value,
                         bool atomic);

  void store_masked(llvm::Value *ptr,
                    llvm::Type *ty,
                    uint64 mask,
                    llvm::Value *value,
                    bool atomic);

  void visit(GlobalStoreStmt *stmt) override;

  llvm::Value *quant_int_or_quant_fixed_to_bits(llvm::Value *val,
                                                Type *input_type,
                                                llvm::Type *output_type);

  void visit(BitStructStoreStmt *stmt) override;

  void store_quant_floats_with_shared_exponents(BitStructStoreStmt *stmt);

  llvm::Value *extract_quant_float(llvm::Value *physical_value,
                                   BitStructType *bit_struct,
                                   int digits_id);

  llvm::Value *extract_quant_int(llvm::Value *physical_value,
                                 llvm::Value *bit_offset,
                                 QuantIntType *qit);

  llvm::Value *reconstruct_quant_fixed(llvm::Value *digits,
                                       QuantFixedType *qfxt);

  llvm::Value *reconstruct_quant_float(llvm::Value *input_digits,
                                       llvm::Value *input_exponent_val,
                                       QuantFloatType *qflt,
                                       bool shared_exponent);

  virtual llvm::Value *create_intrinsic_load(llvm::Value *ptr, llvm::Type *ty);

  void create_global_load(GlobalLoadStmt *stmt, bool should_cache_as_read_only);

  void visit(GlobalLoadStmt *stmt) override;

  void visit(GetRootStmt *stmt) override;

  void visit(BitExtractStmt *stmt) override;

  void visit(LinearizeStmt *stmt) override;

  void visit(IntegerOffsetStmt *stmt) override;

  llvm::Value *create_bit_ptr(llvm::Value *byte_ptr, llvm::Value *bit_offset);

  std::tuple<llvm::Value *, llvm::Value *> load_bit_ptr(llvm::Value *bit_ptr);

  void visit(SNodeLookupStmt *stmt) override;

  void visit(GetChStmt *stmt) override;

  void visit(ExternalPtrStmt *stmt) override;

  void visit(ExternalTensorShapeAlongAxisStmt *stmt) override;

  virtual bool kernel_argument_by_val() const {
    return false;  // on CPU devices just pass in a pointer
  }

  std::string init_offloaded_task_function(OffloadedStmt *stmt,
                                           std::string suffix = "");

  void finalize_offloaded_task_function();

  FunctionCreationGuard get_function_creation_guard(
      std::vector<llvm::Type *> argument_types);

  std::tuple<llvm::Value *, llvm::Value *> get_range_for_bounds(
      OffloadedStmt *stmt);

  virtual void create_offload_range_for(OffloadedStmt *stmt) = 0;

  virtual void create_offload_mesh_for(OffloadedStmt *stmt) {
    TI_NOT_IMPLEMENTED;
  }

  void create_offload_struct_for(OffloadedStmt *stmt, bool spmd = false);

  void visit(LoopIndexStmt *stmt) override;

  void visit(LoopLinearIndexStmt *stmt) override;

  void visit(BlockCornerIndexStmt *stmt) override;

  void visit(GlobalTemporaryStmt *stmt) override;

  void visit(ThreadLocalPtrStmt *stmt) override;

  void visit(BlockLocalPtrStmt *stmt) override;

  void visit(ClearListStmt *stmt) override;

  void visit(InternalFuncStmt *stmt) override;

  // Stack statements

  void visit(AdStackAllocaStmt *stmt) override;

  void visit(AdStackPopStmt *stmt) override;

  void visit(AdStackPushStmt *stmt) override;

  void visit(AdStackLoadTopStmt *stmt) override;

  void visit(AdStackLoadTopAdjStmt *stmt) override;

  void visit(AdStackAccAdjointStmt *stmt) override;

  void visit(RangeAssumptionStmt *stmt) override;

  void visit(LoopUniqueStmt *stmt) override;

  void visit_call_bitcode(ExternalFuncCallStmt *stmt);

  void visit_call_shared_object(ExternalFuncCallStmt *stmt);

  void visit(ExternalFuncCallStmt *stmt) override;

  void visit(MeshPatchIndexStmt *stmt) override;

  void visit(ReferenceStmt *stmt) override;

  void visit(MatrixInitStmt *stmt) override;

  llvm::Value *create_xlogue(std::unique_ptr<Block> &block);

  llvm::Value *create_mesh_xlogue(std::unique_ptr<Block> &block);

  llvm::Value *extract_exponent_from_f32(llvm::Value *f);

  llvm::Value *extract_digits_from_f32(llvm::Value *f, bool full);

  llvm::Value *extract_digits_from_f32_with_shared_exponent(
      llvm::Value *f,
      llvm::Value *shared_exp);

  llvm::Value *get_exponent_offset(llvm::Value *exponent, QuantFloatType *qflt);

  void visit(FuncCallStmt *stmt) override;

  llvm::Value *bitcast_from_u64(llvm::Value *val, DataType type);
  llvm::Value *bitcast_to_u64(llvm::Value *val, DataType type);

  ~TaskCodeGenLLVM() override = default;
};

}  // namespace lang
}  // namespace taichi

#endif  // #ifdef TI_WITH_LLVM
