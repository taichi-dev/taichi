// The LLVM backend for CPUs/NVPTX/AMDGPU
#pragma once

#include <set>
#include <unordered_map>

#include "taichi/ir/ir.h"
#include "taichi/program/program.h"
#include "taichi/llvm/llvm_codegen_utils.h"

TLANG_NAMESPACE_BEGIN

using namespace llvm;

class CodeGenLLVM;

class OffloadedTask {
 public:
  std::string name;
  CodeGenLLVM *codegen;
  using task_fp_type = int32 (*)(void *);
  task_fp_type func;

  int block_dim;
  int grid_dim;

  OffloadedTask(CodeGenLLVM *codegen);

  void begin(const std::string &name);

  void end();

  void compile();

  void operator()(Context *context);
};

class FunctionCreationGuard {
 public:
  CodeGenLLVM *mb;
  llvm::Function *old_func;
  llvm::Function *body;
  llvm::BasicBlock *old_entry, *allocas, *entry;
  llvm::IRBuilder<>::InsertPoint ip;

  FunctionCreationGuard(CodeGenLLVM *mb, std::vector<llvm::Type *> arguments);

  ~FunctionCreationGuard();
};

class CodeGenLLVM : public IRVisitor, public LLVMModuleBuilder {
 public:
  static uint64 task_counter;

  Kernel *kernel;
  IRNode *ir;
  Program *prog;
  std::string kernel_name;
  std::vector<Value *> kernel_args;
  llvm::Type *context_ty;
  llvm::Type *physical_coordinate_ty;
  llvm::Value *current_coordinates;
  // Mainly for supporting continue stmt
  llvm::BasicBlock *current_loop_reentry;
  // Mainly for supporting break stmt
  llvm::BasicBlock *current_while_after_loop;
  llvm::FunctionType *task_function_type;
  OffloadedStmt *current_offloaded_stmt;
  std::unordered_map<Stmt *, llvm::Value *> llvm_val;
  llvm::Function *func;
  std::unique_ptr<OffloadedTask> current_task;
  std::vector<OffloadedTask> offloaded_tasks;
  BasicBlock *func_body_bb;

  std::unordered_map<const Stmt *, std::vector<llvm::Value *>> loop_vars_llvm;

  using IRVisitor::visit;
  using LLVMModuleBuilder::call;

  CodeGenLLVM(Kernel *kernel, IRNode *ir = nullptr);

  Arch current_arch() {
    return kernel->arch;
  }

  void initialize_context();

  llvm::Value *get_arg(int i);

  llvm::Value *get_context();

  llvm::Value *get_root();

  llvm::Value *get_runtime();

  void emit_struct_meta_base(const std::string &name,
                             llvm::Value *node_meta,
                             SNode *snode);

  std::unique_ptr<RuntimeObject> emit_struct_meta_object(SNode *snode);

  llvm::Value *emit_struct_meta(SNode *snode);

  virtual void emit_to_module();

  virtual FunctionType compile_module_to_executable();

  virtual FunctionType gen();

  // only for debugging on CPU
  llvm::Value *create_print(std::string tag, DataType dt, llvm::Value *value);

  llvm::Value *cast_pointer(llvm::Value *val,
                            std::string dest_ty_name,
                            int addr_space = 0);

  void emit_clear_list(OffloadedStmt *listgen);

  void emit_list_gen(OffloadedStmt *listgen);

  void emit_gc(OffloadedStmt *stmt);

  llvm::Value *create_call(llvm::Value *func, std::vector<Value *> args = {});

  llvm::Value *create_call(std::string func_name,
                           std::vector<Value *> args = {});
  llvm::Value *call(SNode *snode,
                    llvm::Value *node_ptr,
                    const std::string &method,
                    const std::vector<llvm::Value *> &arguments);

  void create_increment(llvm::Value *ptr, llvm::Value *value);

  // Direct translation
  void create_naive_range_for(RangeForStmt *for_stmt);

  static std::string get_runtime_snode_name(SNode *snode);

  llvm::Type *llvm_type(DataType dt);

  void visit(Block *stmt_list) override;

  void visit(AllocaStmt *stmt) override;

  void visit(RandStmt *stmt) override;

  virtual void emit_extra_unary(UnaryOpStmt *stmt);

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

  void visit(KernelReturnStmt *stmt) override;

  void visit(LocalLoadStmt *stmt) override;

  void visit(LocalStoreStmt *stmt) override;

  void visit(AssertStmt *stmt) override;

  void visit(SNodeOpStmt *stmt) override;

  void visit(AtomicOpStmt *stmt) override;

  void visit(GlobalPtrStmt *stmt) override;

  void visit(GlobalStoreStmt *stmt) override;

  void visit(GlobalLoadStmt *stmt) override;

  void visit(ElementShuffleStmt *stmt) override;

  void visit(GetRootStmt *stmt) override;

  void visit(OffsetAndExtractBitsStmt *stmt) override;

  void visit(LinearizeStmt *stmt) override;

  void visit(IntegerOffsetStmt *stmt) override;

  void visit(SNodeLookupStmt *stmt) override;

  void visit(GetChStmt *stmt) override;

  void visit(ExternalPtrStmt *stmt) override;

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

  void create_offload_struct_for(OffloadedStmt *stmt, bool spmd = false);

  void visit(LoopIndexStmt *stmt) override;

  void visit(GlobalTemporaryStmt *stmt) override;

  void visit(InternalFuncStmt *stmt) override;

  // Stack statements

  void visit(StackAllocaStmt *stmt) override;

  void visit(StackPopStmt *stmt) override;

  void visit(StackPushStmt *stmt) override;

  void visit(StackLoadTopStmt *stmt) override;

  void visit(StackLoadTopAdjStmt *stmt) override;

  void visit(StackAccAdjointStmt *stmt) override;

  ~CodeGenLLVM() = default;
};

TLANG_NAMESPACE_END
