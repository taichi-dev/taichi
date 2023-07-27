#include <variant>
#include <iostream>
#include <vector>
#include <numeric>

#include "taichi/ir/ir.h"
#include "taichi/ir/statements.h"
#include "taichi/ir/transforms.h"
#include "taichi/ir/visitors.h"
#include "taichi/program/program.h"
#include "taichi/system/profiler.h"

namespace taichi::lang {

class Scalarize : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;
  bool half2_optimization_enabled_ = false;

  explicit Scalarize(IRNode *node, bool half2_optimization)
      : immediate_modifier_(node),
        half2_optimization_enabled_(half2_optimization) {
  }

  /*
    "val" of StoreStmt should have already been replaced by a MatrixInitStmt in
    former scalarization.

    Before:
      StoreStmt(TensorType<4 x i32>* dest, TensorType<4 x i32> val)

    After:
      addr0 = MatrixPtrStmt(TensorType<4 x i32>* dest, 0)
      addr1 = MatrixPtrStmt(TensorType<4 x i32>* dest, 1)
      addr2 = MatrixPtrStmt(TensorType<4 x i32>* dest, 2)
      addr2 = MatrixPtrStmt(TensorType<4 x i32>* dest, 3)

      StoreStmt(i32* addr0, i32 val->cast<MatrixInitStmt>()->val[0])
      StoreStmt(i32* addr1, i32 val->cast<MatrixInitStmt>()->val[1])
      StoreStmt(i32* addr2, i32 val->cast<MatrixInitStmt>()->val[2])
      StoreStmt(i32* addr3, i32 val->cast<MatrixInitStmt>()->val[3])
  */
  template <typename T>
  void scalarize_store_stmt(T *stmt) {
    auto dest_dtype = stmt->dest->ret_type.ptr_removed();
    auto val_dtype = stmt->val->ret_type.ptr_removed();
    if (dest_dtype->template is<TensorType>() &&
        val_dtype->template is<TensorType>()) {
      // Needs scalarize
      auto dest_tensor_type = dest_dtype->template as<TensorType>();
      auto val_tensor_type = val_dtype->template as<TensorType>();

      TI_ASSERT(dest_tensor_type->get_shape() == val_tensor_type->get_shape());

      TI_ASSERT(stmt->val->template is<MatrixInitStmt>());
      auto matrix_init_stmt = stmt->val->template as<MatrixInitStmt>();

      int num_elements = val_tensor_type->get_num_elements();
      auto primitive_type = dest_tensor_type->get_element_type();
      for (int i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        // Merge with previous MatrixPtrStmt
        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(stmt->dest, const_stmt.get());
        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto scalarized_stmt = std::make_unique<T>(matrix_ptr_stmt.get(),
                                                   matrix_init_stmt->values[i]);

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }

      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = LoadStmt(TensorType<4 x i32>* src)

    After:
      i32* addr0 = MatrixPtrStmt(TensorType<4 x i32>* src, 0)
      i32* addr1 = MatrixPtrStmt(TensorType<4 x i32>* src, 1)
      i32* addr2 = MatrixPtrStmt(TensorType<4 x i32>* src, 2)
      i32* addr3 = MatrixPtrStmt(TensorType<4 x i32>* src, 3)

      i32 val0 = LoadStmt(addr0)
      i32 val1 = LoadStmt(addr1)
      i32 val2 = LoadStmt(addr2)
      i32 val3 = LoadStmt(addr3)

      tmp = MatrixInitStmt(val0, val1, val2, val3)
      stmt->replace_all_usages_with(tmp)
  */
  template <typename T>
  void scalarize_load_stmt(T *stmt) {
    auto src_dtype = stmt->src->ret_type.ptr_removed();
    if (src_dtype->template is<TensorType>()) {
      // Needs scalarize
      auto src_tensor_type = src_dtype->template as<TensorType>();

      std::vector<Stmt *> matrix_init_values;
      int num_elements = src_tensor_type->get_num_elements();

      auto primitive_type = src_tensor_type->get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        auto matrix_ptr_stmt =
            std::make_unique<MatrixPtrStmt>(stmt->src, const_stmt.get());

        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        auto scalarized_stmt = std::make_unique<T>(matrix_ptr_stmt.get());
        scalarized_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(scalarized_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(scalarized_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = src_dtype;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
      return;
    }

    // Obtained from Autodiff
    // $2 = adstack top(is_ptr=True)
    // $3 = matrix ptr $2, 0
    // $4 = local load $3

    // during previous irpass::scalarize()
    // $2 = matrix init(...)
    // $3 = matrix ptr $2, 0
    // $4 = local load $3

    // Transform to:
    // $4 = $2->values[0]
    if (stmt->src->template is<MatrixPtrStmt>()) {
      auto matrix_ptr_stmt = stmt->src->template as<MatrixPtrStmt>();
      if (matrix_ptr_stmt->origin->template is<MatrixInitStmt>()) {
        auto matrix_init_stmt =
            matrix_ptr_stmt->origin->template as<MatrixInitStmt>();
        TI_ASSERT(matrix_ptr_stmt->offset->template is<ConstStmt>());
        auto offset_stmt = matrix_ptr_stmt->offset->template as<ConstStmt>();
        int offset = offset_stmt->val.val_int32();

        stmt->replace_usages_with(matrix_init_stmt->values[offset]);
        delayed_modifier_.erase(stmt);
      }
    }
  }

  /*

    Before:
      TensorType<4 x i32> val = UnaryStmt(TensorType<4 x i32> operand)

      * Note that "operand" should have already been scalarized to
    MatrixInitStmt

    After:
      i32 calc_val0 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[0])
      i32 calc_val1 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[1])
      i32 calc_val2 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[2])
      i32 calc_val3 = UnaryStmt(operand->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(calc_val0, calc_val1,
                           calc_val2, calc_val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(UnaryOpStmt *stmt) override {
    auto operand_dtype = stmt->operand->ret_type.ptr_removed();
    auto stmt_dtype = stmt->ret_type.ptr_removed();
    if (operand_dtype->is<TensorType>()) {
      // Needs scalarize
      auto operand_tensor_type = operand_dtype->as<TensorType>();

      TI_ASSERT(stmt->operand->is<MatrixInitStmt>());
      auto operand_matrix_init_stmt = stmt->operand->cast<MatrixInitStmt>();

      TI_ASSERT(operand_matrix_init_stmt->values.size() ==
                operand_tensor_type->get_num_elements());

      std::vector<Stmt *> matrix_init_values;
      int num_elements = operand_tensor_type->get_num_elements();
      auto primitive_type = stmt_dtype.get_element_type();
      for (size_t i = 0; i < num_elements; i++) {
        auto unary_stmt = std::make_unique<UnaryOpStmt>(
            stmt->op_type, operand_matrix_init_stmt->values[i]);
        if (stmt->is_cast()) {
          unary_stmt->cast_type = stmt->cast_type.get_element_type();
        }
        unary_stmt->ret_type = primitive_type;
        matrix_init_values.push_back(unary_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(unary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = operand_dtype;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = BinaryStmt(TensorType<4 x i32> lhs,
                                           TensorType<4 x i32> rhs)

      * Note that "lhs" and "rhs" should have already been scalarized to
    MatrixInitStmt

    After:
      i32 calc_val0 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[0],
                                 rhs->cast<MatrixInitStmt>()->val[0])
      i32 calc_val1 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[1],
                                 rhs->cast<MatrixInitStmt>()->val[1])
      i32 calc_val2 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[2],
                                 rhs->cast<MatrixInitStmt>()->val[2])
      i32 calc_val3 = BinaryStmt(lhs->cast<MatrixInitStmt>()->val[3],
                                 rhs->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(calc_val0, calc_val1,
                           calc_val2, calc_val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(BinaryOpStmt *stmt) override {
    auto lhs_dtype = stmt->lhs->ret_type.ptr_removed();
    auto rhs_dtype = stmt->rhs->ret_type.ptr_removed();
    auto stmt_dtype = stmt->ret_type.ptr_removed();
    if (lhs_dtype->is<TensorType>() || rhs_dtype->is<TensorType>()) {
      // Make sure broadcasting has been correctly applied by
      // BinaryOpExpression::type_check().
      TI_ASSERT(lhs_dtype->is<TensorType>() && rhs_dtype->is<TensorType>());
      // However, since the type conversions are delayed until
      // irpass::type_check(), we only check for the shape here.
      TI_ASSERT(lhs_dtype->cast<TensorType>()->get_shape() ==
                rhs_dtype->cast<TensorType>()->get_shape());
      // Scalarization for LoadStmt should have already replaced both operands
      // to MatrixInitStmt.
      TI_ASSERT(stmt->lhs->is<MatrixInitStmt>());
      TI_ASSERT(stmt->rhs->is<MatrixInitStmt>());

      auto lhs_matrix_init_stmt = stmt->lhs->cast<MatrixInitStmt>();
      std::vector<Stmt *> lhs_vals = lhs_matrix_init_stmt->values;

      auto rhs_matrix_init_stmt = stmt->rhs->cast<MatrixInitStmt>();
      std::vector<Stmt *> rhs_vals = rhs_matrix_init_stmt->values;

      TI_ASSERT(rhs_vals.size() == lhs_vals.size());

      size_t num_elements = lhs_vals.size();
      auto primitive_type = stmt_dtype.get_element_type();
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        auto binary_stmt = std::make_unique<BinaryOpStmt>(
            stmt->op_type, lhs_vals[i], rhs_vals[i]);
        matrix_init_values.push_back(binary_stmt.get());
        binary_stmt->ret_type = primitive_type;

        delayed_modifier_.insert_before(stmt, std::move(binary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
    }
  }

  void visit(PrintStmt *stmt) override {
    auto const &contents = stmt->contents;
    auto const &formats = stmt->formats;

    using EntryType = PrintStmt::EntryType;
    using FormatType = PrintStmt::FormatType;
    using PairType = std::pair<std::vector<EntryType>, std::vector<FormatType>>;
    auto push_content_and_format = [](PairType &pair, EntryType content,
                                      FormatType format = std::nullopt) {
      pair.first.push_back(content);
      pair.second.push_back(format);
    };
    auto get_num_pairs = [](PairType const &pair) -> size_t {
      TI_ASSERT(pair.first.size() == pair.second.size());
      return pair.first.size();
    };
    auto get_pair_at = [](PairType const &pair,
                          size_t index) -> std::pair<EntryType, FormatType> {
      return {pair.first[index], pair.second[index]};
    };

    PairType new_pair;
    for (size_t pair_i = 0; pair_i < contents.size(); ++pair_i) {
      const EntryType &content = contents[pair_i];
      const FormatType &format = formats[pair_i];
      if (auto string_ptr = std::get_if<std::string>(&content)) {
        push_content_and_format(new_pair, *string_ptr, format);
      } else {
        Stmt *print_stmt = std::get<Stmt *>(content);
        if (print_stmt->is<MatrixInitStmt>()) {
          auto matrix_init_stmt = print_stmt->cast<MatrixInitStmt>();
          auto tensor_shape =
              print_stmt->ret_type->as<TensorType>()->get_shape();

          bool is_matrix = tensor_shape.size() == 2;
          int m = tensor_shape[0];

          push_content_and_format(new_pair, "[");
          if (is_matrix) {
            int n = tensor_shape[1];
            for (size_t i = 0; i < m; i++) {
              push_content_and_format(new_pair, "[");
              for (size_t j = 0; j < n; j++) {
                size_t index = i * n + j;
                push_content_and_format(
                    new_pair, matrix_init_stmt->values[index], format);
                if (j != n - 1) {
                  push_content_and_format(new_pair, ", ");
                }
              }
              push_content_and_format(new_pair, "]");

              if (i != m - 1) {
                push_content_and_format(new_pair, ", ");
              }
            }
          } else {
            for (size_t i = 0; i < m; i++) {
              push_content_and_format(new_pair, matrix_init_stmt->values[i],
                                      format);
              if (i != m - 1) {
                push_content_and_format(new_pair, ", ");
              }
            }
          }
          push_content_and_format(new_pair, "]");
        } else {
          push_content_and_format(new_pair, print_stmt, format);
        }
      }
    }

    // Merge string contents
    PairType merged_pair;
    std::string merged_string = "";
    for (size_t i = 0; i < get_num_pairs(new_pair); ++i) {
      auto const &[content, format] = get_pair_at(new_pair, i);
      if (auto string_content = std::get_if<std::string>(&content)) {
        merged_string += *string_content;
        TI_ASSERT(!format.has_value());
      } else {
        if (!merged_string.empty()) {
          push_content_and_format(merged_pair, merged_string);
          merged_string = "";
        }
        push_content_and_format(merged_pair, content, format);
      }
    }
    if (!merged_string.empty()) {
      push_content_and_format(merged_pair, merged_string);
    }

    delayed_modifier_.insert_before(
        stmt, Stmt::make<PrintStmt>(merged_pair.first, merged_pair.second));
    delayed_modifier_.erase(stmt);
  }

  /*
    Before:
      TensorType<4 x i32> val = AtomicStmt(TensorType<4 x i32>* dest,
                                           TensorType<4 x i32>  val)

    After:
      i32* dest_ptr_0 = MatrixPtrStmt(dest, 0)
      i32* dest_ptr_1 = MatrixPtrStmt(dest, 1)
      i32* dest_ptr_2 = MatrixPtrStmt(dest, 2)
      i32* dest_ptr_3 = MatrixPtrStmt(dest, 3)

      i32 dest_val0 = AtomicStmt(dest_ptr_0,
                                 val->cast<MatrixInitStmt>()->val[0])
      i32 dest_val1 = AtomicStmt(dest_ptr_1,
                                 val->cast<MatrixInitStmt>()->val[1])
      i32 dest_val2 = AtomicStmt(dest_ptr_2,
                                 val->cast<MatrixInitStmt>()->val[2])
      i32 dest_val3 = AtomicStmt(dest_ptr_3,
                                 val->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(dest_val0, dest_val1,
                           dest_val2, dest_val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(AtomicOpStmt *stmt) override {
    auto dest_dtype = stmt->dest->ret_type.ptr_removed();
    auto val_dtype = stmt->val->ret_type;

    bool half2_optimizable = half2_optimization_enabled_;
    bool is_tensor_type = false;
    if (dest_dtype->is<TensorType>()) {
      is_tensor_type = true;
      half2_optimizable &=
          (dest_dtype.get_element_type() == PrimitiveType::f16);
      half2_optimizable &=
          (dest_dtype->as<TensorType>()->get_num_elements() == 2);
    } else {
      half2_optimizable = false;
    }
    is_tensor_type |= val_dtype->is<TensorType>();

    if (half2_optimizable) {
      /*
        Before:
          TensorType<2 x i32> old_val = AtomicStmt(TensorType<2 x i32>* dest,
                                                   TensorType<2 x i32>  val)

        After:
          TensorType<2 x i32> old_val = AtomicStmt(TensorType<2 x i32>* dest,
                                                   TensorType<2 x i32>  val)

          TensorType(2, f16)* old_val_alloc = AllocaStmt(TensorType(2, f16))
          LocalStoreStmt(old_val_alloc, old_val)

          f16* old_val_ptr0 = MatrixPtrStmt(old_val_alloc, 0)
          f16* old_val_ptr1 = MatrixPtrStmt(old_val_alloc, 0)

          f16 old_val0 = LoadStmt(old_val_ptr0)
          f16 old_val1 = LoadStmt(old_val_ptr1)

          tmp = MatrixInitStmt(old_val0, old_val1)

          stmt->replace_all_usages_with(tmp)
      */
      auto atomic_stmt =
          std::make_unique<AtomicOpStmt>(stmt->op_type, stmt->dest, stmt->val);
      atomic_stmt->ret_type = stmt->ret_type;

      auto alloca_stmt = std::make_unique<AllocaStmt>(dest_dtype);

      auto local_store_stmt = std::make_unique<LocalStoreStmt>(
          alloca_stmt.get(), atomic_stmt.get());

      auto zero =
          std::make_unique<ConstStmt>(TypedConstant(PrimitiveType::i32, 0));
      auto one =
          std::make_unique<ConstStmt>(TypedConstant(PrimitiveType::i32, 1));

      auto matrix_ptr_0 =
          std::make_unique<MatrixPtrStmt>(alloca_stmt.get(), zero.get());
      auto matrix_ptr_1 =
          std::make_unique<MatrixPtrStmt>(alloca_stmt.get(), one.get());
      matrix_ptr_0->ret_type = PrimitiveType::f16;
      matrix_ptr_0->ret_type.set_is_pointer(true);
      matrix_ptr_1->ret_type = PrimitiveType::f16;
      matrix_ptr_1->ret_type.set_is_pointer(true);

      auto load_stmt_0 = std::make_unique<LocalLoadStmt>(matrix_ptr_0.get());
      auto load_stmt_1 = std::make_unique<LocalLoadStmt>(matrix_ptr_1.get());
      load_stmt_0->ret_type = PrimitiveType::f16;
      load_stmt_1->ret_type = PrimitiveType::f16;

      auto matrix_init_stmt = std::make_unique<MatrixInitStmt>(
          std::vector<Stmt *>{load_stmt_0.get(), load_stmt_1.get()});
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      delayed_modifier_.insert_before(stmt, std::move(alloca_stmt));
      delayed_modifier_.insert_before(stmt, std::move(local_store_stmt));
      delayed_modifier_.insert_before(stmt, std::move(zero));
      delayed_modifier_.insert_before(stmt, std::move(one));
      delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_0));
      delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_1));
      delayed_modifier_.insert_before(stmt, std::move(load_stmt_0));
      delayed_modifier_.insert_before(stmt, std::move(load_stmt_1));
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);

    } else if (is_tensor_type) {
      // Make sure broadcasting has been correctly applied by
      // AtomicOpExpression::type_check().
      TI_ASSERT(dest_dtype->is<TensorType>() && val_dtype->is<TensorType>());
      // However, since the type conversions are delayed until
      // irpass::type_check(), we only check for the shape here.
      TI_ASSERT(dest_dtype->cast<TensorType>()->get_shape() ==
                val_dtype->cast<TensorType>()->get_shape());
      // Scalarization for LoadStmt should have already replaced val operand
      // to MatrixInitStmt.
      TI_ASSERT(stmt->val->is<MatrixInitStmt>());

      auto val_matrix_init_stmt = stmt->val->cast<MatrixInitStmt>();
      std::vector<Stmt *> val_values = val_matrix_init_stmt->values;

      size_t num_elements = val_values.size();
      auto primitive_type = stmt->ret_type.get_element_type();

      // Scalarize dest & val
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        // scalarize to dest_i
        auto const_stmt = std::make_unique<ConstStmt>(
            TypedConstant(get_data_type<int32>(), i));

        // Merge with previous MatrixPtrStmt
        std::unique_ptr<MatrixPtrStmt> matrix_ptr_stmt = nullptr;
        if (stmt->dest->is<MatrixPtrStmt>()) {
          /*
            <*[Tensor (16) [Tensor (4) f32]]> $5 = alloca(shared)
            <*[Tensor (4) f32]> $11 = shift ptr [$5 + $9]
            <[Tensor (4) f32]> $12 : local store [$11 <- $10]
          */
          auto matrix_ptr_stmt_ptr = stmt->dest->as<MatrixPtrStmt>();

          auto base_offset = matrix_ptr_stmt_ptr->offset;
          auto merged_offset = std::make_unique<BinaryOpStmt>(
              BinaryOpType::add, base_offset, const_stmt.get());
          merged_offset->ret_type = base_offset->ret_type;
          matrix_ptr_stmt = std::make_unique<MatrixPtrStmt>(
              matrix_ptr_stmt_ptr->origin, merged_offset.get());
          matrix_ptr_stmt->ret_type = primitive_type;
          matrix_ptr_stmt->ret_type.set_is_pointer(true);

          delayed_modifier_.insert_before(stmt, std::move(merged_offset));
        } else {
          matrix_ptr_stmt =
              std::make_unique<MatrixPtrStmt>(stmt->dest, const_stmt.get());
        }

        matrix_ptr_stmt->ret_type = primitive_type;
        matrix_ptr_stmt->ret_type.set_is_pointer(true);

        // scalarize to val_i
        auto val_stmt = val_values[i];

        // assemble to scalarized atomic_op
        auto atomic_stmt = std::make_unique<AtomicOpStmt>(
            stmt->op_type, matrix_ptr_stmt.get(), val_stmt);
        atomic_stmt->ret_type = primitive_type;

        matrix_init_values.push_back(atomic_stmt.get());

        delayed_modifier_.insert_before(stmt, std::move(const_stmt));
        delayed_modifier_.insert_before(stmt, std::move(matrix_ptr_stmt));
        delayed_modifier_.insert_before(stmt, std::move(atomic_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = TernaryStmt(TensorType<4 x i32> cond,
                                            TensorType<4 x i32> lhs,
                                            TensorType<4 x i32> rhs)

    After:
      i32 val0 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[0],
                             lhs->cast<MatrixInitStmt>()->val[0],
                             rhs->cast<MatrixInitStmt>()->val[0])

      i32 val1 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[1],
                             lhs->cast<MatrixInitStmt>()->val[1],
                             rhs->cast<MatrixInitStmt>()->val[1])

      i32 val2 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[2],
                             lhs->cast<MatrixInitStmt>()->val[2],
                             rhs->cast<MatrixInitStmt>()->val[2])

      i32 val3 = TernaryStmt(cond->cast<MatrixInitStmt>()->val[3],
                             lhs->cast<MatrixInitStmt>()->val[3],
                             rhs->cast<MatrixInitStmt>()->val[3])

      tmp = MatrixInitStmt(val0, val1, val2, val3)

      stmt->replace_all_usages_with(tmp)
  */
  void visit(TernaryOpStmt *stmt) override {
    auto cond_dtype = stmt->op1->ret_type.ptr_removed();
    auto op2_dtype = stmt->op2->ret_type.ptr_removed();
    auto op3_dtype = stmt->op3->ret_type.ptr_removed();
    if (cond_dtype->is<TensorType>()) {
      // Make sure broadcasting has been correctly applied by
      // TernaryOpExpression::type_check().
      TI_ASSERT(cond_dtype->is<TensorType>() && op2_dtype->is<TensorType>() &&
                op3_dtype->is<TensorType>());
      // However, since the type conversions are delayed until
      // irpass::type_check(), we only check for the shape here.
      TI_ASSERT(cond_dtype.get_shape() == op2_dtype.get_shape());
      TI_ASSERT(op2_dtype.get_shape() == op3_dtype.get_shape());
      // Scalarization for LoadStmt should have already replaced all operands
      // to MatrixInitStmt.
      TI_ASSERT(stmt->op1->is<MatrixInitStmt>());
      TI_ASSERT(stmt->op2->is<MatrixInitStmt>());
      TI_ASSERT(stmt->op3->is<MatrixInitStmt>());

      auto cond_matrix_init_stmt = stmt->op1->cast<MatrixInitStmt>();
      std::vector<Stmt *> cond_vals = cond_matrix_init_stmt->values;

      auto op2_matrix_init_stmt = stmt->op2->cast<MatrixInitStmt>();
      std::vector<Stmt *> op2_vals = op2_matrix_init_stmt->values;

      auto op3_matrix_init_stmt = stmt->op3->cast<MatrixInitStmt>();
      std::vector<Stmt *> op3_vals = op3_matrix_init_stmt->values;

      TI_ASSERT(cond_vals.size() == op2_vals.size());
      TI_ASSERT(op2_vals.size() == op3_vals.size());

      size_t num_elements = cond_vals.size();
      auto primitive_type = stmt->ret_type.get_element_type();
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        auto ternary_stmt = std::make_unique<TernaryOpStmt>(
            stmt->op_type, cond_vals[i], op2_vals[i], op3_vals[i]);
        matrix_init_values.push_back(ternary_stmt.get());
        ternary_stmt->ret_type = primitive_type;

        delayed_modifier_.insert_before(stmt, std::move(ternary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
    } else if (cond_dtype->is<PrimitiveType>() &&
               (op2_dtype->is<TensorType>() || op3_dtype->is<TensorType>())) {
      TI_ASSERT(cond_dtype->is<PrimitiveType>() &&
                op2_dtype->is<TensorType>() && op3_dtype->is<TensorType>());
      TI_ASSERT(op2_dtype.get_shape() == op3_dtype.get_shape());
      // Scalarization for LoadStmt should have already replaced all operands
      // to MatrixInitStmt.
      TI_ASSERT(stmt->op2->is<MatrixInitStmt>());
      TI_ASSERT(stmt->op3->is<MatrixInitStmt>());

      Stmt *cond_val = stmt->op1;

      auto op2_matrix_init_stmt = stmt->op2->cast<MatrixInitStmt>();
      std::vector<Stmt *> op2_vals = op2_matrix_init_stmt->values;

      auto op3_matrix_init_stmt = stmt->op3->cast<MatrixInitStmt>();
      std::vector<Stmt *> op3_vals = op3_matrix_init_stmt->values;

      TI_ASSERT(op2_vals.size() == op3_vals.size());

      size_t num_elements = op2_vals.size();
      auto primitive_type = stmt->ret_type.get_element_type();
      std::vector<Stmt *> matrix_init_values;
      for (size_t i = 0; i < num_elements; i++) {
        auto ternary_stmt = std::make_unique<TernaryOpStmt>(
            stmt->op_type, cond_val, op2_vals[i], op3_vals[i]);
        matrix_init_values.push_back(ternary_stmt.get());
        ternary_stmt->ret_type = primitive_type;

        delayed_modifier_.insert_before(stmt, std::move(ternary_stmt));
      }

      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(matrix_init_values);
      matrix_init_stmt->ret_type = stmt->ret_type;

      immediate_modifier_.replace_usages_with(stmt, matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));

      delayed_modifier_.erase(stmt);
    }
  }

  void visit(GlobalStoreStmt *stmt) override {
    scalarize_store_stmt<GlobalStoreStmt>(stmt);
  }

  void visit(LocalStoreStmt *stmt) override {
    scalarize_store_stmt<LocalStoreStmt>(stmt);
  }

  void visit(GlobalLoadStmt *stmt) override {
    scalarize_load_stmt<GlobalLoadStmt>(stmt);
  }

  void visit(LocalLoadStmt *stmt) override {
    scalarize_load_stmt<LocalLoadStmt>(stmt);
  }

  void visit(ArgLoadStmt *stmt) override {
    if (!stmt->ret_type.is_pointer()) {
      return;
    }
    if (stmt->ret_type.ptr_removed()->is<StructType>()) {
      return;
    }
    auto ret_type = stmt->ret_type.ptr_removed().get_element_type();
    ret_type = TypeFactory::get_instance().get_pointer_type(ret_type);
    auto arg_load =
        std::make_unique<ArgLoadStmt>(stmt->arg_id, ret_type, stmt->is_ptr,
                                      stmt->create_load, stmt->arg_depth);

    immediate_modifier_.replace_usages_with(stmt, arg_load.get());

    delayed_modifier_.insert_before(stmt, std::move(arg_load));
    delayed_modifier_.erase(stmt);
  }

  /*
    Split an TensorTyped-AdStack into multiple scalar-typed AdStacks.

    Before:
      TensorType<4 x i32>* stack = AdStackAllocaStmt(TensorType<4 x i32>)

    After:
      i32* stack0 = AdStackAllocaStmt(i32)
      i32* stack1 = AdStackAllocaStmt(i32)
      i32* stack2 = AdStackAllocaStmt(i32)
      i32* stack3 = AdStackAllocaStmt(i32)

      scalarized_ad_stack_map_[stack] = {stack0, stack1, stack2, stack3}
  */
  void visit(AdStackAllocaStmt *stmt) override {
    if (stmt->dt->is<TensorType>()) {
      auto tensor_type = stmt->dt->as<TensorType>();
      auto element_type = tensor_type->get_element_type();
      auto num_elements = tensor_type->get_num_elements();
      std::vector<Stmt *> scalarized_ad_stack;
      for (int i = 0; i < num_elements; i++) {
        auto scalar_ad_stack =
            std::make_unique<AdStackAllocaStmt>(element_type, stmt->max_size);
        scalar_ad_stack->ret_type = element_type;
        scalar_ad_stack->ret_type.set_is_pointer(true);

        scalarized_ad_stack.push_back(scalar_ad_stack.get());
        delayed_modifier_.insert_before(stmt, std::move(scalar_ad_stack));
      }
      scalarized_ad_stack_map_[stmt] = std::move(scalarized_ad_stack);
      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      AdStackPopStmt(TensorType<4 x i32>* stack)

    After:
      AdStackPopStmt(scalarized_ad_stack_map_[stack][0])
      AdStackPopStmt(scalarized_ad_stack_map_[stack][1])
      AdStackPopStmt(scalarized_ad_stack_map_[stack][2])
      AdStackPopStmt(scalarized_ad_stack_map_[stack][3])
  */
  void visit(AdStackPopStmt *stmt) override {
    if (stmt->stack->as<AdStackAllocaStmt>()->dt->is<TensorType>()) {
      auto tensor_type =
          stmt->stack->as<AdStackAllocaStmt>()->dt->as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();
      TI_ASSERT(scalarized_ad_stack_map_.find(stmt->stack) !=
                scalarized_ad_stack_map_.end());
      auto scalarized_ad_stack = scalarized_ad_stack_map_[stmt->stack];
      TI_ASSERT(num_elements == scalarized_ad_stack.size());
      for (int i = 0; i < num_elements; i++) {
        auto scalar_ad_stack_pop =
            std::make_unique<AdStackPopStmt>(scalarized_ad_stack[i]);
        delayed_modifier_.insert_before(stmt, std::move(scalar_ad_stack_pop));
      }
      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = MatrixInitStmt(...) // TI_ASSERT
      AdStackPushStmt(TensorType<4 x i32>* stack, val)

    After:
      AdStackPushStmt(scalarized_ad_stack_map_[stack][0], val->values[0])
      AdStackPushStmt(scalarized_ad_stack_map_[stack][1], val->values[1])
      AdStackPushStmt(scalarized_ad_stack_map_[stack][2], val->values[2])
      AdStackPushStmt(scalarized_ad_stack_map_[stack][3], val->values[3])
  */
  void visit(AdStackPushStmt *stmt) override {
    if (stmt->stack->as<AdStackAllocaStmt>()->dt->is<TensorType>()) {
      auto tensor_type =
          stmt->stack->as<AdStackAllocaStmt>()->dt->as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();
      TI_ASSERT(scalarized_ad_stack_map_.find(stmt->stack) !=
                scalarized_ad_stack_map_.end());
      auto scalarized_ad_stack = scalarized_ad_stack_map_[stmt->stack];
      TI_ASSERT(num_elements == scalarized_ad_stack.size());

      TI_ASSERT(stmt->v->is<MatrixInitStmt>());
      auto matrix_init_stmt = stmt->v->as<MatrixInitStmt>();
      for (int i = 0; i < num_elements; i++) {
        auto scalar_ad_stack_push = std::make_unique<AdStackPushStmt>(
            scalarized_ad_stack[i], matrix_init_stmt->values[i]);
        delayed_modifier_.insert_before(stmt, std::move(scalar_ad_stack_push));
      }
      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      val = AdStackLoadTopStmt(TensorType<4 x i32>* stack)

    After:
      val0 = AdStackLoadTopStmt(scalarized_ad_stack_map_[stack][0])
      val1 = AdStackLoadTopStmt(scalarized_ad_stack_map_[stack][1])
      val2 = AdStackLoadTopStmt(scalarized_ad_stack_map_[stack][2])
      val3 = AdStackLoadTopStmt(scalarized_ad_stack_map_[stack][3])

      matrix_init_stmt = MatrixInitStmt({val0, val1, val2, val3})

      replace_all_usages_with(val, matrix_init_stmt)
  */
  void visit(AdStackLoadTopStmt *stmt) override {
    if (stmt->stack->as<AdStackAllocaStmt>()->dt->is<TensorType>()) {
      auto tensor_type =
          stmt->stack->as<AdStackAllocaStmt>()->dt->as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();
      auto element_type = tensor_type->get_element_type();
      TI_ASSERT(scalarized_ad_stack_map_.find(stmt->stack) !=
                scalarized_ad_stack_map_.end());
      auto scalarized_ad_stack = scalarized_ad_stack_map_[stmt->stack];
      TI_ASSERT(num_elements == scalarized_ad_stack.size());

      std::vector<Stmt *> scalar_ad_stack_load_top;
      for (int i = 0; i < num_elements; i++) {
        auto scalar_ad_stack_load_top_stmt =
            std::make_unique<AdStackLoadTopStmt>(scalarized_ad_stack[i]);
        scalar_ad_stack_load_top_stmt->ret_type = element_type;

        scalar_ad_stack_load_top.push_back(scalar_ad_stack_load_top_stmt.get());
        delayed_modifier_.insert_before(
            stmt, std::move(scalar_ad_stack_load_top_stmt));
      }
      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(scalar_ad_stack_load_top);
      matrix_init_stmt->ret_type = tensor_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      val = AdStackLoadTopAdjStmt(TensorType<4 x i32>* stack)

    After:
      val0 = AdStackLoadTopAdjStmt(scalarized_ad_stack_map_[stack][0])
      val1 = AdStackLoadTopAdjStmt(scalarized_ad_stack_map_[stack][1])
      val2 = AdStackLoadTopAdjStmt(scalarized_ad_stack_map_[stack][2])
      val3 = AdStackLoadTopAdjStmt(scalarized_ad_stack_map_[stack][3])

      matrix_init_stmt = MatrixInitStmt({val0, val1, val2, val3})

      replace_all_usages_with(val, matrix_init_stmt)
  */
  void visit(AdStackLoadTopAdjStmt *stmt) override {
    if (stmt->stack->as<AdStackAllocaStmt>()->dt->is<TensorType>()) {
      auto tensor_type =
          stmt->stack->as<AdStackAllocaStmt>()->dt->as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();
      auto element_type = tensor_type->get_element_type();
      TI_ASSERT(scalarized_ad_stack_map_.find(stmt->stack) !=
                scalarized_ad_stack_map_.end());
      auto scalarized_ad_stack = scalarized_ad_stack_map_[stmt->stack];
      TI_ASSERT(num_elements == scalarized_ad_stack.size());

      std::vector<Stmt *> scalar_ad_stack_load_top;
      for (int i = 0; i < num_elements; i++) {
        auto scalar_ad_stack_load_top_stmt =
            std::make_unique<AdStackLoadTopAdjStmt>(scalarized_ad_stack[i]);
        scalar_ad_stack_load_top_stmt->ret_type = element_type;

        scalar_ad_stack_load_top.push_back(scalar_ad_stack_load_top_stmt.get());
        delayed_modifier_.insert_before(
            stmt, std::move(scalar_ad_stack_load_top_stmt));
      }
      auto matrix_init_stmt =
          std::make_unique<MatrixInitStmt>(scalar_ad_stack_load_top);
      matrix_init_stmt->ret_type = tensor_type;

      stmt->replace_usages_with(matrix_init_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(matrix_init_stmt));
      delayed_modifier_.erase(stmt);
    }
  }

  /*
    Before:
      TensorType<4 x i32> val = MatrixInitStmt(...) // TI_ASSERT
      AdStackAccAdjointStmt(TensorType<4 x i32>* stack, val)

    After:
      AdStackAccAdjointStmt(scalarized_ad_stack_map_[stack][0], val->values[0])
      AdStackAccAdjointStmt(scalarized_ad_stack_map_[stack][1], val->values[1])
      AdStackAccAdjointStmt(scalarized_ad_stack_map_[stack][2], val->values[2])
      AdStackAccAdjointStmt(scalarized_ad_stack_map_[stack][3], val->values[3])
  */
  void visit(AdStackAccAdjointStmt *stmt) override {
    if (stmt->stack->as<AdStackAllocaStmt>()->dt->is<TensorType>()) {
      auto tensor_type =
          stmt->stack->as<AdStackAllocaStmt>()->dt->as<TensorType>();
      auto num_elements = tensor_type->get_num_elements();
      TI_ASSERT(scalarized_ad_stack_map_.find(stmt->stack) !=
                scalarized_ad_stack_map_.end());
      auto scalarized_ad_stack = scalarized_ad_stack_map_[stmt->stack];
      TI_ASSERT(num_elements == scalarized_ad_stack.size());

      TI_ASSERT(stmt->v->is<MatrixInitStmt>());
      auto matrix_init_stmt = stmt->v->as<MatrixInitStmt>();
      for (int i = 0; i < num_elements; i++) {
        auto scalar_ad_stack_push = std::make_unique<AdStackAccAdjointStmt>(
            scalarized_ad_stack[i], matrix_init_stmt->values[i]);
        delayed_modifier_.insert_before(stmt, std::move(scalar_ad_stack_push));
      }
      delayed_modifier_.erase(stmt);
    }
  }

  static bool run(IRNode *node, bool half2_optimization_enabled) {
    Scalarize pass(node, half2_optimization_enabled);
    node->accept(&pass);
    return pass.delayed_modifier_.modify_ir();
  }

 private:
  using BasicStmtVisitor::visit;
  std::unordered_map<Stmt *, std::vector<Stmt *>> scalarized_ad_stack_map_;
};

// The GatherScalarizableLocalPointers gathers all local TensorType allocas
// only indexed with constants, which can then be scalarized in the
// ScalarizeLocalPointers pass.
class GatherScalarizableLocalPointers : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;

  std::unordered_map<Stmt *, bool> is_alloca_scalarizable_;

 public:
  void visit(AllocaStmt *stmt) override {
    if (stmt->ret_type.ptr_removed()->is<TensorType>()) {
      TI_ASSERT(is_alloca_scalarizable_.count(stmt) == 0);
      is_alloca_scalarizable_[stmt] = !stmt->is_shared;
    }
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AllocaStmt>()) {
      TI_ASSERT(is_alloca_scalarizable_.count(stmt->origin) == 1);
      if (!stmt->offset->is<ConstStmt>()) {
        is_alloca_scalarizable_[stmt->origin] = false;
      }
    }
  }

  static std::unordered_set<Stmt *> run(IRNode *node) {
    GatherScalarizableLocalPointers pass;
    node->accept(&pass);
    std::unordered_set<Stmt *> result;
    for (auto &[k, v] : pass.is_alloca_scalarizable_) {
      if (v) {
        result.insert(k);
      }
    }
    return result;
  }
};

class ScalarizePointers : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  std::unordered_set<Stmt *> scalarizable_allocas_;
  // { original_alloca_stmt : [scalarized_alloca_stmt0, ...] }
  std::unordered_map<Stmt *, std::vector<Stmt *>> scalarized_local_tensor_map_;

  explicit ScalarizePointers(
      IRNode *node,
      const std::unordered_set<Stmt *> &scalarizable_allocas)
      : immediate_modifier_(node), scalarizable_allocas_(scalarizable_allocas) {
  }

  /*
    Accessing scalar values are always more efficient than accessing elements
    from a vector - the former generates less instructions, leading to better
    performance in both compilation and runtime.

    Although we can do nothing about "global" tensors like tensors from
    ArgLoadStmt or GlobalPtrStmt, we can still optimize "local" tensors like
    tensors from AllocaStmt. In this pass, we ask AllocaStmt to allocate
    multiple scalarized PrimitiveTyped variables in replacement of the
    original TensorType.

    An additional container "scalarized_local_tensor_map_" is used to keep
    track of the scalarized AllocaStmt, for later use in LoadStmt and
    StoreStmt.

    Before:
      TensorType<4 x i32>* addr = AllocaStmt(TensorType<4 x i32>)

    After:
      i32 addr0 = AllocaStmt(i32)
      i32 addr1 = AllocaStmt(i32)
      i32 addr2 = AllocaStmt(i32)
      i32 addr3 = AllocaStmt(i32)

      scalarized_local_tensor_map_[addr] = {addr0, addr1, addr2, addr3}
  */
  void visit(AllocaStmt *stmt) override {
    if (scalarizable_allocas_.count(stmt) == 1) {
      auto tensor_type = stmt->ret_type.ptr_removed()->cast<TensorType>();
      TI_ASSERT(tensor_type != nullptr);
      auto primitive_type = tensor_type->get_element_type();

      TI_ASSERT(scalarized_local_tensor_map_.count(stmt) == 0);
      scalarized_local_tensor_map_[stmt] = {};
      for (size_t i = 0; i < tensor_type->get_num_elements(); i++) {
        auto scalarized_alloca_stmt =
            std::make_unique<AllocaStmt>(primitive_type);
        scalarized_alloca_stmt->ret_type =
            TypeFactory::get_instance().get_pointer_type(primitive_type);

        scalarized_local_tensor_map_[stmt].push_back(
            scalarized_alloca_stmt.get());
        delayed_modifier_.insert_before(stmt,
                                        std::move(scalarized_alloca_stmt));
      }

      delayed_modifier_.erase(stmt);
    }
  }

  void visit(MatrixPtrStmt *stmt) override {
    /*
      Before:
        MatrixPtrStmt(TensorType<4 x i32>* alloca_stmt, int offset)

      After:
        scalarized_alloca_stmt =
      scalarized_local_tensor_map_[alloca_stmt][offset]
        stmt->replace_all_usages_with(scalarized_alloca_stmt)
    */
    if (stmt->origin->is<AllocaStmt>() &&
        scalarizable_allocas_.count(stmt->origin) == 1) {
      auto alloca_stmt = stmt->origin->cast<AllocaStmt>();
      auto tensor_type =
          alloca_stmt->ret_type.ptr_removed()->cast<TensorType>();
      TI_ASSERT(tensor_type != nullptr);
      int num_elements = tensor_type->get_num_elements();
      TI_ASSERT(scalarized_local_tensor_map_.count(alloca_stmt));

      const auto &scalarized_alloca_stmts =
          scalarized_local_tensor_map_[alloca_stmt];
      TI_ASSERT(scalarized_alloca_stmts.size() == num_elements);

      TI_ASSERT(stmt->offset->is<ConstStmt>());
      int offset = stmt->offset->cast<ConstStmt>()->val.val_int32();

      TI_ASSERT(offset < scalarized_alloca_stmts.size());
      auto new_stmt = scalarized_alloca_stmts[offset];

      immediate_modifier_.replace_usages_with(stmt, new_stmt);
      delayed_modifier_.erase(stmt);
      return;
    }

    /*
      Before:
        TensorType<4 x i32>* ptr = GlobalTempStmt(offset_0)
        i32* ptr_1 = MatrixPtrStmt(ptr, offset_1)

      After:
        i32* $1 = GlobalTempStmt(offset_0 + offset_1 * sizeof(i32))
        replace_all_usages_with(ptr_1, $1)
    */
    if (stmt->origin->is<GlobalTemporaryStmt>() &&
        stmt->offset->is<ConstStmt>()) {
      auto global_temp_stmt = stmt->origin->as<GlobalTemporaryStmt>();
      auto offset_0 = global_temp_stmt->offset;
      auto offset_1 = stmt->offset->as<ConstStmt>()->val.val_int32();
      auto new_offset =
          offset_0 + offset_1 * data_type_size(stmt->ret_type.ptr_removed());

      auto new_global_temp_stmt = std::make_unique<GlobalTemporaryStmt>(
          new_offset, stmt->ret_type.ptr_removed().get_element_type());
      new_global_temp_stmt->ret_type.set_is_pointer(true);

      stmt->replace_usages_with(new_global_temp_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(new_global_temp_stmt));
      delayed_modifier_.erase(stmt);
      return;
    }

    /*
      Before:
        TensorType<4 x i32>* ptr = ThreadLocalPtrStmt(offset_0)
        i32* ptr_1 = MatrixPtrStmt(ptr, offset_1)

      After:
        i32* $1 = ThreadLocalPtrStmt(offset_0 + offset_1 * sizeof(i32))
        replace_all_usages_with(ptr_1, $1)
    */
    if (stmt->origin->is<ThreadLocalPtrStmt>() &&
        stmt->offset->is<ConstStmt>()) {
      auto thread_local_stmt = stmt->origin->as<ThreadLocalPtrStmt>();
      auto offset_0 = thread_local_stmt->offset;
      auto offset_1 = stmt->offset->as<ConstStmt>()->val.val_int32();
      auto new_offset =
          offset_0 + offset_1 * data_type_size(stmt->ret_type.ptr_removed());

      auto new_thread_local_stmt = std::make_unique<ThreadLocalPtrStmt>(
          new_offset, stmt->ret_type.ptr_removed().get_element_type());
      new_thread_local_stmt->ret_type.set_is_pointer(true);

      stmt->replace_usages_with(new_thread_local_stmt.get());
      delayed_modifier_.insert_before(stmt, std::move(new_thread_local_stmt));
      delayed_modifier_.erase(stmt);
      return;
    }
  }

  static bool run(IRNode *node,
                  const std::unordered_set<Stmt *> &scalarizable_allocas) {
    ScalarizePointers pass(node, scalarizable_allocas);
    node->accept(&pass);
    return pass.delayed_modifier_.modify_ir();
  }

 private:
  using BasicStmtVisitor::visit;
};

// The ExtractLocalPointers pass aims at removing redundant ConstStmts and
// MatrixPtrStmts generated for any (AllocaStmt, integer) pair by extracting
// a unique copy for any future usage.
//
// Example for redundant stmts:
//   <i32> $0 = const 0
//   <i32> $1 = const 1
//   ...
//   <[Tensor (3, 3) f32]> $47738 = alloca
//   <i32> $47739 = const 0  [REDUNDANT]
//   <*f32> $47740 = shift ptr [$47738 + $47739]
//   $47741 : local store [$47740 <- $47713]
//   <i32> $47742 = const 1  [REDUNDANT]
//   <*f32> $47743 = shift ptr [$47738 + $47742]
//   $47744 : local store [$47743 <- $47716]
//   ...
//   <i32> $47812 = const 1  [REDUNDANT]
//   <*f32> $47813 = shift ptr [$47738 + $47812]  [REDUNDANT]
//   <f32> $47814 = local load [$47813]
class ExtractLocalPointers : public BasicStmtVisitor {
 public:
  ImmediateIRModifier immediate_modifier_;
  DelayedIRModifier delayed_modifier_;

  std::unordered_map<std::pair<Stmt *, int>,
                     Stmt *,
                     hashing::Hasher<std::pair<Stmt *, int>>>
      first_matrix_ptr_;  // mapping an (AllocaStmt, integer) pair to the
                          // first MatrixPtrStmt representing it
  std::unordered_map<int, Stmt *>
      first_const_;  // mapping an integer to the first ConstStmt representing
                     // it
  Block *top_level_;

  explicit ExtractLocalPointers(IRNode *root) : immediate_modifier_(root) {
    if (root->is<OffloadedStmt>()) {
      top_level_ = root->as<OffloadedStmt>()->body.get();
    } else {
      TI_ASSERT(root->is<Block>());
      top_level_ = root->as<Block>();
    }
  }

  void visit(OffloadedStmt *stmt) override {
    // Extract to OffloadStmt
    Block *orig_top_level = top_level_;
    top_level_ = stmt->body.get();
    stmt->all_blocks_accept(this);
    top_level_ = orig_top_level;
  }

  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<AllocaStmt>()) {
      auto alloca_stmt = stmt->origin->cast<AllocaStmt>();
      auto tensor_type =
          alloca_stmt->ret_type.ptr_removed()->cast<TensorType>();
      TI_ASSERT(tensor_type != nullptr);
      if (stmt->offset->is<ConstStmt>()) {
        int offset = stmt->offset->cast<ConstStmt>()->val.val_int32();
        if (first_const_.count(offset) == 0) {
          first_const_[offset] = stmt->offset;
          delayed_modifier_.extract_to_block_front(stmt->offset, top_level_);
        }
        auto key = std::make_pair(alloca_stmt, offset);
        if (first_matrix_ptr_.count(key) == 0) {
          auto extracted = std::make_unique<MatrixPtrStmt>(
              alloca_stmt, first_const_[offset]);
          first_matrix_ptr_[key] = extracted.get();
          delayed_modifier_.insert_after(alloca_stmt, std::move(extracted));
        }
        auto new_stmt = first_matrix_ptr_[key];
        immediate_modifier_.replace_usages_with(stmt, new_stmt);
        delayed_modifier_.erase(stmt);
      }
    }
  }

  static bool run(IRNode *node) {
    ExtractLocalPointers pass(node);
    node->accept(&pass);
    return pass.delayed_modifier_.modify_ir();
  }

 private:
  using BasicStmtVisitor::visit;
};

class FuseMatrixPtr : public BasicStmtVisitor {
 private:
  using BasicStmtVisitor::visit;
  DelayedIRModifier modifier_;

 public:
  void visit(MatrixPtrStmt *stmt) override {
    if (stmt->origin->is<ExternalPtrStmt>()) {
      auto origin = stmt->origin->as<ExternalPtrStmt>();
      TI_ASSERT(stmt->origin->ret_type.ptr_removed()->is<TensorType>());

      std::vector<Stmt *> indices = origin->indices;
      indices.push_back(stmt->offset);

      // MatrixPtrStmt has flattened indices, linearization of which is done
      // during IndexExpression::flatten() Here we need to modify the
      // element_dim and element_shape a little bit.
      std::vector<int> element_shape = {
          std::accumulate(begin(origin->element_shape),
                          end(origin->element_shape), 1, std::multiplies<>())};

      auto fused = std::make_unique<ExternalPtrStmt>(
          origin->base_ptr, indices, origin->ndim, element_shape,
          origin->is_grad);
      fused->ret_type = stmt->ret_type;
      // Note: Update base_ptr's ret_type so that it matches the ExternalPtrStmt
      // with flattened indices. Main goal is to keep all the hacks in a single
      // place so that they're easier to remove
      auto members = origin->base_ptr->as<ArgLoadStmt>()
                         ->ret_type.ptr_removed()
                         ->as<StructType>()
                         ->elements();
      bool needs_grad = members.size() > TypeFactory::GRAD_PTR_POS_IN_NDARRAY;
      auto type = TypeFactory::get_instance().get_ndarray_struct_type(
          fused->ret_type.ptr_removed(), origin->ndim, needs_grad);
      origin->base_ptr->as<ArgLoadStmt>()->ret_type =
          TypeFactory::get_instance().get_pointer_type((Type *)type);
      stmt->replace_usages_with(fused.get());
      modifier_.insert_before(stmt, std::move(fused));
      modifier_.erase(stmt);
      return;
    }

    if (stmt->origin->is<GetChStmt>() &&
        stmt->origin->ret_type.ptr_removed()->is<TensorType>()) {
      auto origin = stmt->origin->as<GetChStmt>();

      if (!stmt->offset->is<ConstStmt>()) {
        return;
      }

      int offset = stmt->offset->as<ConstStmt>()->val.val_int32();

      auto input_ptr = origin->input_ptr;
      auto input_snode = origin->input_snode;
      bool is_bit_vectorized = origin->is_bit_vectorized;

      auto new_get_ch_stmt = std::make_unique<GetChStmt>(
          input_ptr, input_snode, origin->chid + offset, is_bit_vectorized);
      new_get_ch_stmt->ret_type = stmt->ret_type;

      stmt->replace_usages_with(new_get_ch_stmt.get());
      modifier_.insert_before(stmt, std::move(new_get_ch_stmt));
      modifier_.erase(stmt);
      return;
    }
  }

  static bool run(IRNode *node) {
    FuseMatrixPtr pass;
    node->accept(&pass);
    return pass.modifier_.modify_ir();
  }
};

namespace irpass {

bool scalarize(IRNode *root, bool half2_optimization_enabled) {
  TI_AUTO_PROF;
  bool modified = false;

  modified |= Scalarize::run(root, half2_optimization_enabled);
  auto scalarizable_allocas = GatherScalarizableLocalPointers::run(root);
  modified |= ScalarizePointers::run(root, scalarizable_allocas);
  modified |= ExtractLocalPointers::run(root);
  modified |= FuseMatrixPtr::run(root);
  return modified;
}

}  // namespace irpass

}  // namespace taichi::lang
