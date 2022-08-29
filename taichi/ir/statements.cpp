// TODO: gradually cppize statements.h
#include "taichi/ir/statements.h"
#include "taichi/util/bit.h"

TLANG_NAMESPACE_BEGIN

UnaryOpStmt::UnaryOpStmt(UnaryOpType op_type, Stmt *operand)
    : op_type(op_type), operand(operand) {
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
                                 const std::vector<Stmt *> &indices)
    : base_ptr(base_ptr), indices(indices) {
  TI_ASSERT(base_ptr != nullptr);
  TI_ASSERT(base_ptr->is<ArgLoadStmt>());
  TI_STMT_REG_FIELDS;
}

ExternalPtrStmt::ExternalPtrStmt(Stmt *base_ptr,
                                 const std::vector<Stmt *> &indices,
                                 const std::vector<int> &element_shape,
                                 int element_dim)
    : ExternalPtrStmt(base_ptr, indices) {
  this->element_shape = element_shape;
  this->element_dim = element_dim;
}

GlobalPtrStmt::GlobalPtrStmt(SNode *snode,
                             const std::vector<Stmt *> &indices,
                             bool activate)
    : snode(snode),
      indices(indices),
      activate(activate),
      is_bit_vectorized(false) {
  TI_ASSERT(snode != nullptr);
  element_type() = snode->dt;
  TI_STMT_REG_FIELDS;
}

PtrOffsetStmt::PtrOffsetStmt(Stmt *origin_input, Stmt *offset_input) {
  origin = origin_input;
  offset = offset_input;
  if (origin->is<AllocaStmt>()) {
    TI_ASSERT(origin->cast<AllocaStmt>()->ret_type->is<TensorType>());
    auto tensor_type = origin->cast<AllocaStmt>()->ret_type->cast<TensorType>();
    element_type() = tensor_type->get_element_type();
    element_type().set_is_pointer(true);
  } else if (origin->is<GlobalTemporaryStmt>()) {
    TI_ASSERT(origin->cast<GlobalTemporaryStmt>()->ret_type->is<TensorType>());
    auto tensor_type =
        origin->cast<GlobalTemporaryStmt>()->ret_type->cast<TensorType>();
    element_type() = tensor_type->get_element_type();
    element_type().set_is_pointer(true);
  } else if (origin->is<GlobalPtrStmt>()) {
    element_type() = origin->cast<GlobalPtrStmt>()->ret_type;
  } else {
    TI_ERROR(
        "PtrOffsetStmt must be used for AllocaStmt / GlobalTemporaryStmt "
        "(locally) or GlobalPtrStmt (globally).")
  }
  TI_STMT_REG_FIELDS;
}

SNodeOpStmt::SNodeOpStmt(SNodeOpType op_type,
                         SNode *snode,
                         Stmt *ptr,
                         Stmt *val)
    : op_type(op_type), snode(snode), ptr(ptr), val(val) {
  element_type() = PrimitiveType::i32;
  TI_STMT_REG_FIELDS;
}

bool SNodeOpStmt::activation_related(SNodeOpType op) {
  return op == SNodeOpType::activate || op == SNodeOpType::deactivate ||
         op == SNodeOpType::is_active;
}

bool SNodeOpStmt::need_activation(SNodeOpType op) {
  return op == SNodeOpType::activate || op == SNodeOpType::append;
}

ExternalTensorShapeAlongAxisStmt::ExternalTensorShapeAlongAxisStmt(int axis,
                                                                   int arg_id)
    : axis(axis), arg_id(arg_id) {
  TI_STMT_REG_FIELDS;
}

LoopUniqueStmt::LoopUniqueStmt(Stmt *input, const std::vector<SNode *> &covers)
    : input(input) {
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

IfStmt::IfStmt(Stmt *cond) : cond(cond) {
  TI_STMT_REG_FIELDS;
}

void IfStmt::set_true_statements(std::unique_ptr<Block> &&new_true_statements) {
  true_statements = std::move(new_true_statements);
  if (true_statements)
    true_statements->parent_stmt = this;
}

void IfStmt::set_false_statements(
    std::unique_ptr<Block> &&new_false_statements) {
  false_statements = std::move(new_false_statements);
  if (false_statements)
    false_statements->parent_stmt = this;
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
  this->body->parent_stmt = this;
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
  this->body->parent_stmt = this;
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
  this->body->parent_stmt = this;
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

WhileStmt::WhileStmt(std::unique_ptr<Block> &&body)
    : mask(nullptr), body(std::move(body)) {
  this->body->parent_stmt = this;
  TI_STMT_REG_FIELDS;
}

std::unique_ptr<Stmt> WhileStmt::clone() const {
  auto new_stmt = std::make_unique<WhileStmt>(body->clone());
  new_stmt->mask = mask;
  return new_stmt;
}

GetChStmt::GetChStmt(Stmt *input_ptr, int chid, bool is_bit_vectorized)
    : input_ptr(input_ptr), chid(chid), is_bit_vectorized(is_bit_vectorized) {
  TI_ASSERT(input_ptr->is<SNodeLookupStmt>());
  input_snode = input_ptr->as<SNodeLookupStmt>()->snode;
  output_snode = input_snode->ch[chid].get();
  TI_STMT_REG_FIELDS;
}

OffloadedStmt::OffloadedStmt(TaskType task_type, Arch arch)
    : task_type(task_type), device(arch) {
  if (has_body()) {
    body = std::make_unique<Block>();
    body->parent_stmt = this;
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
  auto new_stmt = std::make_unique<OffloadedStmt>(task_type, device);
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
    new_stmt->tls_prologue->parent_stmt = new_stmt.get();
  }
  if (mesh_prologue) {
    new_stmt->mesh_prologue = mesh_prologue->clone();
    new_stmt->mesh_prologue->parent_stmt = new_stmt.get();
  }
  if (bls_prologue) {
    new_stmt->bls_prologue = bls_prologue->clone();
    new_stmt->bls_prologue->parent_stmt = new_stmt.get();
  }
  if (body) {
    new_stmt->body = body->clone();
    new_stmt->body->parent_stmt = new_stmt.get();
  }
  if (bls_epilogue) {
    new_stmt->bls_epilogue = bls_epilogue->clone();
    new_stmt->bls_epilogue->parent_stmt = new_stmt.get();
  }
  if (tls_epilogue) {
    new_stmt->tls_epilogue = tls_epilogue->clone();
    new_stmt->tls_epilogue->parent_stmt = new_stmt.get();
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

int LoopIndexStmt::max_num_bits() const {
  if (auto range_for = loop->cast<RangeForStmt>()) {
    // Return the max number of bits only if both begin and end are
    // non-negative consts.
    if (!range_for->begin->is<ConstStmt>() || !range_for->end->is<ConstStmt>())
      return -1;
    auto begin = range_for->begin->as<ConstStmt>();
    if (begin->val.val_int() < 0)
      return -1;
    auto end = range_for->end->as<ConstStmt>();
    return (int)bit::ceil_log2int(end->val.val_int());
  } else if (auto struct_for = loop->cast<StructForStmt>()) {
    return struct_for->snode->get_num_bits(index);
  } else if (auto offload = loop->cast<OffloadedStmt>()) {
    if (offload->task_type == OffloadedStmt::TaskType::range_for) {
      if (!offload->const_begin || !offload->const_end)
        return -1;
      if (offload->begin_value < 0)
        return -1;
      return bit::ceil_log2int(offload->end_value);
    } else if (offload->task_type == OffloadedStmt::TaskType::struct_for) {
      return offload->snode->get_num_bits(index);
    } else {
      TI_NOT_IMPLEMENTED
    }
  } else {
    TI_NOT_IMPLEMENTED
  }
}

BitStructType *BitStructStoreStmt::get_bit_struct() const {
  return ptr->as<SNodeLookupStmt>()->snode->dt->as<BitStructType>();
}

TLANG_NAMESPACE_END
