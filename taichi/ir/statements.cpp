// TODO: gradually cppize statements.h
#include "taichi/ir/statements.h"
#include "taichi/util/bit.h"
#include "taichi/program/kernel.h"
#include "taichi/program/function.h"

namespace taichi::lang {

UnaryOpStmt::UnaryOpStmt(UnaryOpType op_type,
                         Stmt *operand,
                         const DebugInfo &dbg_info)
    : Stmt(dbg_info), op_type(op_type), operand(operand) {
  TI_ASSERT(!operand->is<AllocaStmt>());
  cast_type = PrimitiveType::unknown;
  TI_STMT_REG_FIELDS;
}

DecorationStmt::DecorationStmt(Stmt *operand,
                               const std::vector<uint32_t> &decoration)
    : operand(operand), decoration(decoration) {
  TI_STMT_REG_FIELDS;
}

bool UnaryOpStmt::is_cast() const {
  return unary_op_is_cast(op_type);
}

bool UnaryOpStmt::same_operation(UnaryOpStmt *o) const {
  if (op_type == o->op_type) {
    if (is_cast()) {
      return cast_type == o->cast_type;
    } else {
      return true;
    }
  }
  return false;
}

ExternalPtrStmt::ExternalPtrStmt(Stmt *base_ptr,
                                 const std::vector<Stmt *> &indices,
                                 bool is_grad,
                                 BoundaryMode boundary)
    : base_ptr(base_ptr),
      indices(indices),
      is_grad(is_grad),
      boundary(boundary) {
  ndim = indices.size();
  TI_ASSERT(base_ptr != nullptr);
  TI_ASSERT(base_ptr->is<ArgLoadStmt>());
  TI_STMT_REG_FIELDS;
}

ExternalPtrStmt::ExternalPtrStmt(Stmt *base_ptr,
                                 const std::vector<Stmt *> &indices,
                                 int ndim,
                                 const std::vector<int> &element_shape,
                                 bool is_grad,
                                 BoundaryMode boundary)
    : ExternalPtrStmt(base_ptr, indices, is_grad, boundary) {
  this->element_shape = element_shape;
  this->ndim = ndim;
}

GlobalPtrStmt::GlobalPtrStmt(SNode *snode,
                             const std::vector<Stmt *> &indices,
                             bool activate,
                             bool is_cell_access,
                             const DebugInfo &dbg_info)
    : Stmt(dbg_info),
      snode(snode),
      indices(indices),
      activate(activate),
      is_cell_access(is_cell_access),
      is_bit_vectorized(false) {
  TI_ASSERT(snode != nullptr);
  element_type() = snode->dt;
  TI_STMT_REG_FIELDS;
}

MatrixOfGlobalPtrStmt::MatrixOfGlobalPtrStmt(const std::vector<SNode *> &snodes,
                                             const std::vector<Stmt *> &indices,
                                             bool dynamic_indexable,
                                             int dynamic_index_stride,
                                             DataType dt,
                                             bool activate)
    : snodes(snodes),
      indices(indices),
      dynamic_indexable(dynamic_indexable),
      dynamic_index_stride(dynamic_index_stride),
      activate(activate) {
  ret_type = dt;
  TI_STMT_REG_FIELDS;
}

MatrixOfMatrixPtrStmt::MatrixOfMatrixPtrStmt(const std::vector<Stmt *> &stmts,
                                             DataType dt)
    : stmts(stmts) {
  ret_type = dt;
  ret_type.set_is_pointer(true);
  TI_STMT_REG_FIELDS;
}

MatrixPtrStmt::MatrixPtrStmt(Stmt *origin_input,
                             Stmt *offset_input,
                             const DebugInfo &dbg_info) {
  origin = origin_input;
  offset = offset_input;
  this->dbg_info = dbg_info;

  if (origin->is<AllocaStmt>() || origin->is<GlobalTemporaryStmt>() ||
      origin->is<ExternalPtrStmt>() || origin->is<MatrixOfGlobalPtrStmt>() ||
      origin->is<MatrixOfMatrixPtrStmt>() || origin->is<ThreadLocalPtrStmt>() ||
      origin->is<MatrixPtrStmt>()) {
    auto tensor_type = origin->ret_type.ptr_removed()->cast<TensorType>();
    TI_ASSERT(tensor_type != nullptr);
    element_type() = tensor_type->get_element_type();
    element_type().set_is_pointer(true);
  } else if (origin->is<GlobalPtrStmt>() || origin->is<GetChStmt>()) {
    element_type() = origin->ret_type.ptr_removed().get_element_type();
    element_type().set_is_pointer(true);
  } else if (origin->is<AdStackLoadTopStmt>()) {
    TI_ASSERT(origin->as<AdStackLoadTopStmt>()->return_ptr == true);
    element_type() = origin->ret_type.get_element_type();
    element_type().set_is_pointer(true);
  } else {
    TI_ERROR(
        "MatrixPtrStmt must be used for AllocaStmt / GlobalTemporaryStmt "
        "(locally) or GlobalPtrStmt / MatrixOfGlobalPtrStmt / ExternalPtrStmt "
        "(globally).")
  }
  TI_STMT_REG_FIELDS;
}

bool MatrixPtrStmt::common_statement_eliminable() const {
  Callable *callable = get_callable();
  TI_ASSERT(callable != nullptr);
  return (callable->autodiff_mode == AutodiffMode::kNone);
}

SNodeOpStmt::SNodeOpStmt(SNodeOpType op_type,
                         SNode *snode,
                         Stmt *ptr,
                         Stmt *val,
                         const DebugInfo &dbg_info)
    : Stmt(dbg_info), op_type(op_type), snode(snode), ptr(ptr), val(val) {
  element_type() = PrimitiveType::i32;
  TI_STMT_REG_FIELDS;
}

bool SNodeOpStmt::activation_related(SNodeOpType op) {
  return op == SNodeOpType::activate || op == SNodeOpType::deactivate ||
         op == SNodeOpType::is_active;
}

bool SNodeOpStmt::need_activation(SNodeOpType op) {
  return op == SNodeOpType::activate || op == SNodeOpType::append ||
         op == SNodeOpType::allocate;
}

ExternalTensorShapeAlongAxisStmt::ExternalTensorShapeAlongAxisStmt(
    int axis,
    const std::vector<int> &arg_id,
    const DebugInfo &dbg_info)
    : Stmt(dbg_info), axis(axis), arg_id(arg_id) {
  TI_STMT_REG_FIELDS;
}

ExternalTensorBasePtrStmt::ExternalTensorBasePtrStmt(
    const std::vector<int> &arg_id,
    bool is_grad,
    const DebugInfo &dbg_info)
    : Stmt(dbg_info), arg_id(arg_id), is_grad(is_grad) {
  TI_STMT_REG_FIELDS;
}

LoopUniqueStmt::LoopUniqueStmt(Stmt *input,
                               const std::vector<SNode *> &covers,
                               const DebugInfo &dbg_info)
    : Stmt(dbg_info), input(input) {
  for (const auto &sn : covers) {
    if (sn->is_place()) {
      TI_INFO(
          "A place SNode {} appears in the 'covers' parameter "
          "of 'ti.loop_unique'. It is recommended to use its parent "
          "(x.parent()) instead.",
          sn->get_node_type_name_hinted());
      this->covers.insert(sn->parent->id);
    } else
      this->covers.insert(sn->id);
  }
  TI_STMT_REG_FIELDS;
}

IfStmt::IfStmt(Stmt *cond, const DebugInfo &dbg_info)
    : Stmt(dbg_info), cond(cond) {
  TI_STMT_REG_FIELDS;
}

void IfStmt::set_true_statements(std::unique_ptr<Block> &&new_true_statements) {
  true_statements = std::move(new_true_statements);
  if (true_statements)
    true_statements->set_parent_stmt(this);
}

void IfStmt::set_false_statements(
    std::unique_ptr<Block> &&new_false_statements) {
  false_statements = std::move(new_false_statements);
  if (false_statements)
    false_statements->set_parent_stmt(this);
}

std::unique_ptr<Stmt> IfStmt::clone() const {
  auto new_stmt = std::make_unique<IfStmt>(cond);
  if (true_statements)
    new_stmt->set_true_statements(true_statements->clone());
  if (false_statements)
    new_stmt->set_false_statements(false_statements->clone());
  return new_stmt;
}

RangeForStmt::RangeForStmt(Stmt *begin,
                           Stmt *end,
                           std::unique_ptr<Block> &&body,
                           bool is_bit_vectorized,
                           int num_cpu_threads,
                           int block_dim,
                           bool strictly_serialized,
                           std::string range_hint)
    : begin(begin),
      end(end),
      body(std::move(body)),
      is_bit_vectorized(is_bit_vectorized),
      num_cpu_threads(num_cpu_threads),
      block_dim(block_dim),
      strictly_serialized(strictly_serialized),
      range_hint(range_hint) {
  reversed = false;
  this->body->set_parent_stmt(this);
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> RangeForStmt::clone() const {
  auto new_stmt = std::make_unique<RangeForStmt>(
      begin, end, body->clone(), is_bit_vectorized, num_cpu_threads, block_dim,
      strictly_serialized);
  new_stmt->reversed = reversed;
  return new_stmt;
}

StructForStmt::StructForStmt(SNode *snode,
                             std::unique_ptr<Block> &&body,
                             bool is_bit_vectorized,
                             int num_cpu_threads,
                             int block_dim)
    : snode(snode),
      body(std::move(body)),
      is_bit_vectorized(is_bit_vectorized),
      num_cpu_threads(num_cpu_threads),
      block_dim(block_dim) {
  this->body->set_parent_stmt(this);
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> StructForStmt::clone() const {
  auto new_stmt = std::make_unique<StructForStmt>(
      snode, body->clone(), is_bit_vectorized, num_cpu_threads, block_dim);
  new_stmt->mem_access_opt = mem_access_opt;
  return new_stmt;
}

MeshForStmt::MeshForStmt(mesh::Mesh *mesh,
                         mesh::MeshElementType element_type,
                         std::unique_ptr<Block> &&body,
                         bool is_bit_vectorized,
                         int num_cpu_threads,
                         int block_dim)
    : mesh(mesh),
      body(std::move(body)),
      is_bit_vectorized(is_bit_vectorized),
      num_cpu_threads(num_cpu_threads),
      block_dim(block_dim),
      major_from_type(element_type) {
  this->body->set_parent_stmt(this);
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> MeshForStmt::clone() const {
  auto new_stmt = std::make_unique<MeshForStmt>(
      mesh, major_from_type, body->clone(), is_bit_vectorized, num_cpu_threads,
      block_dim);
  new_stmt->major_to_types = major_to_types;
  new_stmt->minor_relation_types = minor_relation_types;
  new_stmt->mem_access_opt = mem_access_opt;
  return new_stmt;
}

FuncCallStmt::FuncCallStmt(Function *func, const std::vector<Stmt *> &args)
    : func(func), args(args) {
  TI_STMT_REG_FIELDS;
}

stmt_refs FuncCallStmt::get_store_destination() const {
  std::vector<Stmt *> ret;
  for (auto &arg : args) {
    if (auto ref = arg->cast<ReferenceStmt>()) {
      ret.push_back(ref->var);
    } else if (arg->ret_type.is_pointer()) {
      ret.push_back(arg);
    }
  }
  ret.insert(ret.end(), func->store_dests.begin(), func->store_dests.end());
  return ret;
}

WhileStmt::WhileStmt(std::unique_ptr<Block> &&body)
    : mask(nullptr), body(std::move(body)) {
  this->body->set_parent_stmt(this);
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> WhileStmt::clone() const {
  auto new_stmt = std::make_unique<WhileStmt>(body->clone());
  new_stmt->mask = mask;
  return new_stmt;
}

GetChStmt::GetChStmt(Stmt *input_ptr,
                     int chid,
                     bool is_bit_vectorized,
                     const DebugInfo &dbg_info)
    : Stmt(dbg_info),
      input_ptr(input_ptr),
      chid(chid),
      is_bit_vectorized(is_bit_vectorized) {
  TI_ASSERT(input_ptr->is<SNodeLookupStmt>());
  input_snode = input_ptr->as<SNodeLookupStmt>()->snode;
  output_snode = input_snode->ch[chid].get();
  TI_STMT_REG_FIELDS;
}

GetChStmt::GetChStmt(Stmt *input_ptr,
                     SNode *snode,
                     int chid,
                     bool is_bit_vectorized,
                     const DebugInfo &dbg_info)
    : Stmt(dbg_info),
      input_ptr(input_ptr),
      chid(chid),
      is_bit_vectorized(is_bit_vectorized) {
  input_snode = snode;
  output_snode = input_snode->ch[chid].get();
  TI_STMT_REG_FIELDS;
}

OffloadedStmt::OffloadedStmt(TaskType task_type, Arch arch, Kernel *kernel)
    : kernel_(kernel), task_type(task_type), device(arch) {
  if (has_body()) {
    body = std::make_unique<Block>();
    body->set_parent_stmt(this);
  }
  TI_STMT_REG_FIELDS;
}

std::string OffloadedStmt::task_name() const {
  if (task_type == TaskType::serial) {
    return "serial";
  } else if (task_type == TaskType::range_for) {
    return "range_for";
  } else if (task_type == TaskType::struct_for) {
    return "struct_for";
  } else if (task_type == TaskType::mesh_for) {
    return "mesh_for";
  } else if (task_type == TaskType::listgen) {
    TI_ASSERT(snode);
    return fmt::format("listgen_{}", snode->get_node_type_name_hinted());
  } else if (task_type == TaskType::gc) {
    TI_ASSERT(snode);
    return fmt::format("gc_{}", snode->name);
  } else {
    TI_NOT_IMPLEMENTED
  }
}

// static
std::string OffloadedStmt::task_type_name(TaskType tt) {
  return offloaded_task_type_name(tt);
}

std::unique_ptr<Stmt> OffloadedStmt::clone() const {
  auto new_stmt = std::make_unique<OffloadedStmt>(task_type, device, kernel_);
  new_stmt->snode = snode;
  new_stmt->begin_offset = begin_offset;
  new_stmt->end_offset = end_offset;
  new_stmt->const_begin = const_begin;
  new_stmt->const_end = const_end;
  new_stmt->begin_value = begin_value;
  new_stmt->end_value = end_value;
  new_stmt->grid_dim = grid_dim;
  new_stmt->block_dim = block_dim;
  new_stmt->reversed = reversed;
  new_stmt->is_bit_vectorized = is_bit_vectorized;
  new_stmt->num_cpu_threads = num_cpu_threads;
  new_stmt->index_offsets = index_offsets;

  new_stmt->mesh = mesh;
  new_stmt->major_from_type = major_from_type;
  new_stmt->major_to_types = major_to_types;
  new_stmt->minor_relation_types = minor_relation_types;

  new_stmt->owned_offset_local = owned_offset_local;
  new_stmt->total_offset_local = total_offset_local;
  new_stmt->owned_num_local = owned_num_local;
  new_stmt->total_num_local = total_num_local;

  if (tls_prologue) {
    new_stmt->tls_prologue = tls_prologue->clone();
    new_stmt->tls_prologue->set_parent_stmt(new_stmt.get());
  }
  if (mesh_prologue) {
    new_stmt->mesh_prologue = mesh_prologue->clone();
    new_stmt->mesh_prologue->set_parent_stmt(new_stmt.get());
  }
  if (bls_prologue) {
    new_stmt->bls_prologue = bls_prologue->clone();
    new_stmt->bls_prologue->set_parent_stmt(new_stmt.get());
  }
  if (body) {
    new_stmt->body = body->clone();
    new_stmt->body->set_parent_stmt(new_stmt.get());
  }
  if (bls_epilogue) {
    new_stmt->bls_epilogue = bls_epilogue->clone();
    new_stmt->bls_epilogue->set_parent_stmt(new_stmt.get());
  }
  if (tls_epilogue) {
    new_stmt->tls_epilogue = tls_epilogue->clone();
    new_stmt->tls_epilogue->set_parent_stmt(new_stmt.get());
  }
  new_stmt->tls_size = tls_size;
  new_stmt->bls_size = bls_size;
  new_stmt->mem_access_opt = mem_access_opt;
  return new_stmt;
}

void OffloadedStmt::all_blocks_accept(IRVisitor *visitor,
                                      bool skip_mesh_prologue) {
  if (tls_prologue)
    tls_prologue->accept(visitor);
  if (mesh_prologue && !skip_mesh_prologue)
    mesh_prologue->accept(visitor);
  if (bls_prologue)
    bls_prologue->accept(visitor);
  if (body)
    body->accept(visitor);
  if (bls_epilogue)
    bls_epilogue->accept(visitor);
  if (tls_epilogue)
    tls_epilogue->accept(visitor);
}

bool is_clear_list_task(const OffloadedStmt *stmt) {
  return (stmt->task_type == OffloadedStmt::TaskType::serial) &&
         (stmt->body->size() == 1) && stmt->body->back()->is<ClearListStmt>();
}

ClearListStmt::ClearListStmt(SNode *snode) : snode(snode) {
  TI_STMT_REG_FIELDS;
}

BitStructType *BitStructStoreStmt::get_bit_struct() const {
  return ptr->as<SNodeLookupStmt>()->snode->dt->as<BitStructType>();
}

}  // namespace taichi::lang
