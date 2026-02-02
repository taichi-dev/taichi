#include "taichi/codegen/flagos/codegen_flagos.h"

#include <vector>
#include <set>
#include <functional>

#include "taichi/common/core.h"
#include "taichi/util/io.h"
#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/program/program.h"
#include "taichi/util/lang_util.h"
#include "taichi/rhi/flagos/flagos_device.h"
#include "taichi/runtime/program_impls/llvm/llvm_program.h"
#include "taichi/analysis/offline_cache_util.h"
#include "taichi/ir/analysis.h"
#include "taichi/ir/transforms.h"
#include "taichi/codegen/codegen_utils.h"

namespace taichi {
namespace lang {

using namespace llvm;

void TaskCodeGenFlagOS::create_print(std::string tag,
                                     DataType dt,
                                     llvm::Value *value){
    // FlagOS print functionality
    // TODO: Implement using FlagOS debug/logging API
    TI_NOT_IMPLEMENTED}

std::tuple<llvm::Value *, llvm::Type *> TaskCodeGenFlagOS::
    create_value_and_type(llvm::Value *value, DataType dt) {
  TI_NOT_IMPLEMENTED
}

void TaskCodeGenFlagOS::visit(PrintStmt *stmt) {
  // FlagOS print support
  // TODO: Implement using FlagOS debug API
  TI_NOT_IMPLEMENTED
}

void TaskCodeGenFlagOS::emit_extra_unary(UnaryOpStmt *stmt) {
  auto input = llvm_val[stmt->operand];
  auto input_taichi_type = stmt->operand->ret_type;
  auto op = stmt->op_type;

  // FlagOS uses standard LLVM math intrinsics or device math library
  // For now, delegate to base class implementation
  TaskCodeGenLLVM::emit_extra_unary(stmt);

  // TODO: Add FlagOS-specific optimizations
  // e.g., use fast math intrinsics for AI chips
}

llvm::Value *TaskCodeGenFlagOS::optimized_reduction(AtomicOpStmt *stmt) {
  if (!stmt->is_reduction) {
    return nullptr;
  }
  TI_ASSERT(stmt->val->ret_type->is<PrimitiveType>());
  PrimitiveTypeID prim_type = stmt->val->ret_type->cast<PrimitiveType>()->type;

  // FlagOS reduction operations
  // These can be optimized using chip-specific instructions
  std::unordered_map<PrimitiveTypeID,
                     std::unordered_map<AtomicOpType, std::string>>
      fast_reductions;

  fast_reductions[PrimitiveTypeID::i32][AtomicOpType::add] =
      "flagos_reduce_add_i32";
  fast_reductions[PrimitiveTypeID::f32][AtomicOpType::add] =
      "flagos_reduce_add_f32";
  fast_reductions[PrimitiveTypeID::i32][AtomicOpType::min] =
      "flagos_reduce_min_i32";
  fast_reductions[PrimitiveTypeID::f32][AtomicOpType::min] =
      "flagos_reduce_min_f32";
  fast_reductions[PrimitiveTypeID::i32][AtomicOpType::max] =
      "flagos_reduce_max_i32";
  fast_reductions[PrimitiveTypeID::f32][AtomicOpType::max] =
      "flagos_reduce_max_f32";

  fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_and] =
      "flagos_reduce_and_i32";
  fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_or] =
      "flagos_reduce_or_i32";
  fast_reductions[PrimitiveTypeID::i32][AtomicOpType::bit_xor] =
      "flagos_reduce_xor_i32";

  AtomicOpType op = stmt->op_type;
  if (fast_reductions.find(prim_type) == fast_reductions.end()) {
    return nullptr;
  }
  TI_ASSERT(fast_reductions.at(prim_type).find(op) !=
            fast_reductions.at(prim_type).end());
  return call(fast_reductions.at(prim_type).at(op),
              {llvm_val[stmt->dest], llvm_val[stmt->val]});
}

void TaskCodeGenFlagOS::visit(RangeForStmt *for_stmt) {
  create_naive_range_for(for_stmt);
}

void TaskCodeGenFlagOS::create_offload_range_for(OffloadedStmt *stmt) {
  auto tls_prologue = create_xlogue(stmt->tls_prologue);

  llvm::Function *body;
  {
    auto guard = get_function_creation_guard(
        {llvm::PointerType::get(get_runtime_type("RuntimeContext"), 0),
         llvm::PointerType::get(get_runtime_type("Element"), 0),
         get_tls_buffer_type(), tlctx->get_data_type<int>()});

    auto loop_var = create_entry_block_alloca(PrimitiveType::i32);
    loop_vars_llvm[stmt].push_back(loop_var);
    builder->CreateStore(get_arg(3), loop_var);
    stmt->body->accept(this);

    body = guard.body;
  }

  auto epilogue = create_xlogue(stmt->tls_epilogue);

  auto [begin, end] = get_range_for_bounds(stmt);
  call("gpu_parallel_range_for",
       {get_arg(0), begin, end, tls_prologue, body, epilogue,
        tlctx->get_constant(stmt->tls_size)});
}

void TaskCodeGenFlagOS::create_offload_mesh_for(OffloadedStmt *stmt) {
  TI_NOT_IMPLEMENTED
}

void TaskCodeGenFlagOS::emit_flagos_gc(OffloadedStmt *stmt) {
  auto snode_id = tlctx->get_constant(stmt->snode->id);
  {
    init_offloaded_task_function(stmt, "gather_list");
    call("gc_parallel_0", get_context(), snode_id);
    finalize_offloaded_task_function();
    current_task->grid_dim = compile_config.saturating_grid_dim;
    current_task->block_dim = 64;
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
  }
  {
    init_offloaded_task_function(stmt, "reinit_lists");
    call("gc_parallel_1", get_context(), snode_id);
    finalize_offloaded_task_function();
    current_task->grid_dim = 1;
    current_task->block_dim = 1;
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
  }
  {
    init_offloaded_task_function(stmt, "zero_fill");
    call("gc_parallel_2", get_context(), snode_id);
    finalize_offloaded_task_function();
    current_task->grid_dim = compile_config.saturating_grid_dim;
    current_task->block_dim = 64;
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
  }
}

void TaskCodeGenFlagOS::create_bls_buffer(OffloadedStmt *stmt) {
  TI_NOT_IMPLEMENTED
}

void TaskCodeGenFlagOS::visit(OffloadedStmt *stmt) {
  if (stmt->bls_size > 0)
    create_bls_buffer(stmt);

#if defined(TI_WITH_FLAGOS)
  TI_ASSERT(current_offload == nullptr);
  current_offload = stmt;
  using Type = OffloadedStmt::TaskType;
  if (stmt->task_type == Type::gc) {
    emit_flagos_gc(stmt);
  } else {
    init_offloaded_task_function(stmt);
    if (stmt->task_type == Type::serial) {
      stmt->body->accept(this);
    } else if (stmt->task_type == Type::range_for) {
      create_offload_range_for(stmt);
    } else if (stmt->task_type == Type::struct_for) {
      create_offload_struct_for(stmt);
    } else if (stmt->task_type == Type::mesh_for) {
      create_offload_mesh_for(stmt);
    } else if (stmt->task_type == Type::listgen) {
      emit_list_gen(stmt);
    } else {
      TI_NOT_IMPLEMENTED
    }
    finalize_offloaded_task_function();

    // Configure grid and block dimensions for FlagOS
    current_task->grid_dim = stmt->grid_dim;
    if (stmt->task_type == Type::range_for) {
      if (stmt->const_begin && stmt->const_end) {
        int num_threads = stmt->end_value - stmt->begin_value;
        int grid_dim = ((num_threads % stmt->block_dim) == 0)
                           ? (num_threads / stmt->block_dim)
                           : (num_threads / stmt->block_dim) + 1;
        grid_dim = std::max(grid_dim, 1);
        current_task->grid_dim = std::min(stmt->grid_dim, grid_dim);
      }
    }
    if (stmt->task_type == Type::listgen) {
      // Use device-specific block configuration
      // TODO: Query FlagOS for optimal configuration
      int num_compute_units = 64;  // Placeholder
      current_task->grid_dim = num_compute_units * 4;
    }
    current_task->block_dim = stmt->block_dim;
    TI_ASSERT(current_task->grid_dim != 0);
    TI_ASSERT(current_task->block_dim != 0);
    offloaded_tasks.push_back(*current_task);
    current_task = nullptr;
  }
  current_offload = nullptr;
#else
  TI_NOT_IMPLEMENTED
#endif
}

void TaskCodeGenFlagOS::visit(ExternalFuncCallStmt *stmt) {
  if (stmt->type == ExternalFuncCallStmt::BITCODE) {
    TaskCodeGenLLVM::visit_call_bitcode(stmt);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

void TaskCodeGenFlagOS::visit(BinaryOpStmt *stmt) {
  auto op = stmt->op_type;
  auto ret_taichi_type = stmt->ret_type;

  // Delegate to base class for standard operations
  TaskCodeGenLLVM::visit(stmt);

  // TODO: Add FlagOS-specific optimizations for math operations
  // e.g., use fast approximations for transcendental functions
}

std::tuple<llvm::Value *, llvm::Value *> TaskCodeGenFlagOS::get_spmd_info() {
  // Get SPMD (Single Program Multiple Data) execution information
  // This is similar to CUDA/AMDGPU but uses FlagOS abstractions

  // TODO: Use FlagOS intrinsics when available
  // For now, use LLVM GPU intrinsics
  auto thread_idx =
      builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_tid_x, {}, {});
  auto block_dim =
      builder->CreateIntrinsic(Intrinsic::nvvm_read_ptx_sreg_ntid_x, {}, {});

  return std::make_tuple(thread_idx, block_dim);
}

LLVMCompiledTask KernelCodeGenFlagOS::compile_task(
    int task_codegen_id,
    const CompileConfig &config,
    std::unique_ptr<llvm::Module> &&module,
    IRNode *block) {
  TaskCodeGenFlagOS gen(task_codegen_id, config, get_taichi_llvm_context(),
                        kernel, block);
  return gen.run_compilation();
}

}  // namespace lang

}  // namespace taichi
