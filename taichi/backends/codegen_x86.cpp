// x86 backend implementation

#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../tlang_util.h"
#include "codegen_x86.h"
#include "loopgen.h"
#include "../program.h"
#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class CPUIRCodeGen : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  CodeGenBase *codegen;
  LoopGenerator loopgen;
  Kernel *kernel;
  std::unique_ptr<Stmt> atomic_add;

  CPUIRCodeGen(CodeGenBase *codegen) : codegen(codegen), loopgen(codegen) {
    current_struct_for = nullptr;
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    codegen->emit(f, std::forward<Args>(args)...);
  }

  static void run(CodeGenBase *codegen, IRNode *node, Kernel *kernel) {
    auto p = CPUIRCodeGen(codegen);
    p.kernel = kernel;
    node->accept(&p);
  }

  void visit(Block *stmt_list) {
    for (auto &stmt : stmt_list->statements) {
      stmt->accept(this);
    }
  }

  void visit(AllocaStmt *alloca) {
    emit("{} {}(0);", alloca->ret_data_type_name(), alloca->raw_name());
  }

  void visit(RandStmt *stmt) {
    TC_ASSERT(stmt->ret_type.data_type == DataType::f32);
    emit("const auto {} = {}::rand();", stmt->raw_name(),
         stmt->ret_data_type_name());
  }

  void visit(UnaryOpStmt *stmt) {
    if (stmt->op_type != UnaryOpType::cast) {
      emit("const {} {}({}({}));", stmt->ret_data_type_name(), stmt->raw_name(),
           unary_op_type_name(stmt->op_type), stmt->operand->raw_name());
    } else {
      if (stmt->cast_by_value) {
        emit("const {} {}(cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->operand->raw_name());
      } else {
        emit("const {} {}(union_cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->operand->raw_name());
      }
    }
  }

  void visit(BinaryOpStmt *bin) {
    emit("const {} {}({}({}, {}));", bin->ret_data_type_name(), bin->raw_name(),
         binary_op_type_name(bin->op_type), bin->lhs->raw_name(),
         bin->rhs->raw_name());
  }

  void visit(TernaryOpStmt *tri) {
    emit("const {} {}({}({}, {}, {}));", tri->ret_data_type_name(),
         tri->raw_name(), ternary_type_name(tri->op_type), tri->op1->raw_name(),
         tri->op2->raw_name(), tri->op3->raw_name());
  }

  void visit(IfStmt *if_stmt) {
    if (if_stmt->true_statements) {
      emit("if (any({})) {{", if_stmt->true_mask->raw_name());
      if_stmt->true_statements->accept(this);
      emit("}}");
    }
    if (if_stmt->false_statements) {
      emit("if (any({})) {{", if_stmt->false_mask->raw_name());
      if_stmt->false_statements->accept(this);
      emit("}}");
    }
  }

  void visit(PrintStmt *print_stmt) {
    if (print_stmt->width() == 1) {
      emit("std::cout << \"[debug] \" \"{}\" \" = \" << {} << std::endl;",
           print_stmt->str, print_stmt->stmt->raw_name());
    } else {
      emit(R"(std::cout << "[debug] {} = "; {}.print();)", print_stmt->str,
           print_stmt->stmt->raw_name());
    }
  }

  void visit(ConstStmt *const_stmt) {
    emit("const {} {}({});", const_stmt->ret_type.str(), const_stmt->raw_name(),
         const_stmt->val.serialize(
             [&](const TypedConstant &t) { return t.stringify(); }, "{"));
  }

  void visit(WhileControlStmt *stmt) {
    emit("{} = bit_and({}, {});", stmt->mask->raw_name(),
         stmt->mask->raw_name(), stmt->cond->raw_name());
    emit("if (!any({})) break;", stmt->mask->raw_name());
  }

  void visit(WhileStmt *stmt) {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(StructForStmt *for_stmt) {
    TC_ASSERT_INFO(current_struct_for == nullptr,
                   "Struct for cannot be nested.");
    current_struct_for = for_stmt;
    emit("{{");
    auto leaf = for_stmt->snode->parent;

    loopgen.loop_gen_leaves(for_stmt, leaf);

    std::string vars;
    for (int i = 0; i < (int)for_stmt->loop_vars.size(); i++) {
      vars += for_stmt->loop_vars[i]->raw_name();
      if (i + 1 < (int)for_stmt->loop_vars.size()) {
        vars += ",";
      }
    }
    emit("int num_leaves = leaves.size();");
    if (for_stmt->parallelize) {
      emit("omp_set_num_threads({});", for_stmt->parallelize);
      emit("#pragma omp parallel for schedule(dynamic) private({})", vars);
    }
    emit("for (int leaf_loop = 0; leaf_loop < num_leaves; leaf_loop++) {{");

    if (kernel->is_reduction) {
      atomic_add = std::move(for_stmt->body->statements.back());
      for_stmt->body->statements.resize((int)for_stmt->body->statements.size() -
                                        1);
      // initialize back
      auto atomic = atomic_add->as<AtomicOpStmt>();
      auto ptr = atomic->dest;
      emit("{} reduction(0);", ptr->ret_data_type_name());
    }
    loopgen.emit_load_from_context(leaf);
    loopgen.generate_single_loop_header(leaf, true, for_stmt->vectorize);
    loopgen.emit_setup_loop_variables(for_stmt, leaf);
    for_stmt->body->accept(this);
    if (kernel->is_reduction) {
      auto atomic = atomic_add->as<AtomicOpStmt>();
      emit("reduction = add(reduction, {});", atomic->val->raw_name());
    }
    emit("}}");
    if (kernel->is_reduction) {
      // write back
      emit("for (int i = 1; i < {}; i++) {{", atomic_add->width());
      emit("reduction[0] += reduction[i];");
      emit("}}");
      auto atomic = atomic_add->as<AtomicOpStmt>();
      std::string node_type_name;
      if (atomic->dest->is<GlobalPtrStmt>()) {
        auto ptr = atomic->dest->as<GlobalPtrStmt>();
        node_type_name = ptr->snodes[0]->node_type_name;
      } else if (atomic->dest->is<ElementShuffleStmt>()) {
        node_type_name = atomic->dest->as<ElementShuffleStmt>()
                             ->elements[0]
                             .stmt->as<GetChStmt>()
                             ->output_snode->node_type_name;
      } else {
        node_type_name =
            atomic->dest->as<GetChStmt>()->output_snode->node_type_name;
      }
      emit("atomic_add(&access_{}(root, 0, 0, 0, 0)->val, reduction[0]);",
           node_type_name);
    }
    emit("}}");
    emit("}}");
    current_struct_for = nullptr;
  }

  void visit(RangeForStmt *for_stmt) {
    auto loop_var = for_stmt->loop_var;
    if (for_stmt->parallelize) {
      // emit("#pragma omp parallel for num_threads({})",
      // for_stmt->parallelize);
      emit("omp_set_num_threads({});", for_stmt->parallelize);
      emit("#pragma omp parallel for schedule(dynamic) private({})",
           loop_var->raw_name());
    }
    if (loop_var->ret_type.width == 1 &&
        loop_var->ret_type.data_type == DataType::i32) {
      if (!for_stmt->reversed) {
        // normal loop
        emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), for_stmt->vectorize);
        emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      } else {
        // reversed loop
        TC_ASSERT(for_stmt->vectorize == 1);
        emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), for_stmt->vectorize);
        emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      }
    } else {
      emit("for ({} {} = {}; {} < {}; {} = {} + {}({})) {{",
           loop_var->ret_data_type_name(), loop_var->raw_name(),
           for_stmt->begin->raw_name(), loop_var->raw_name(),
           for_stmt->end->raw_name(), loop_var->raw_name(),
           loop_var->raw_name(), loop_var->ret_data_type_name(),
           for_stmt->vectorize);
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(ArgLoadStmt *stmt) {
    if (stmt->is_ptr) {
      auto dt = data_type_name(stmt->ret_type.data_type);
      emit("const {} * {}(context.get_arg<{} *>({}));", dt, stmt->raw_name(),
           dt, stmt->arg_id);
    } else {
      emit("const {} {}({{context.get_arg<{}>({})}});",
           stmt->ret_data_type_name(), stmt->raw_name(),
           data_type_name(stmt->ret_type.data_type), stmt->arg_id);
    }
  }

  void visit(LocalLoadStmt *stmt) {
    // TODO: optimize for partially vectorized load...

    bool linear_index = true;
    for (int i = 0; i < (int)stmt->ptr.size(); i++) {
      if (stmt->ptr[i].offset != i) {
        linear_index = false;
      }
    }
    if (stmt->same_source() && linear_index &&
        stmt->width() == stmt->ptr[0].var->width()) {
      auto ptr = stmt->ptr[0].var;
      emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
           ptr->raw_name());
    } else {
      std::string init_v;
      for (int i = 0; i < stmt->width(); i++) {
        init_v += fmt::format("{}[{}]", stmt->ptr[i].var->raw_name(),
                              stmt->ptr[i].offset);
        if (i + 1 < stmt->width()) {
          init_v += ", ";
        }
      }
      emit("const {} {}({{{}}});", stmt->ret_data_type_name(), stmt->raw_name(),
           init_v);
    }
  }

  void visit(LocalStoreStmt *stmt) {
    auto mask = stmt->parent->mask();
    if (mask) {
      emit("{} = select({}, {}, {});", stmt->ptr->raw_name(), mask->raw_name(),
           stmt->data->raw_name(), stmt->ptr->raw_name());
    } else {
      emit("{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
    }
  }

  void visit(SNodeOpStmt *stmt) {
    stmt->ret_type.data_type = DataType::i32;
    if (stmt->op_type == SNodeOpType::probe) {
      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
    }

    for (auto l = 0; l < stmt->width(); l++) {
      auto snode = stmt->snodes[l];
      auto indices = indices_str(snode, l, stmt->indices);

      emit("{{");
      if (stmt->op_type != SNodeOpType::activate &&
          stmt->op_type != SNodeOpType::probe) {
        emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
             snode->node_type_name, snode->node_type_name,
             make_list(indices, ""));
      }
      if (stmt->op_type == SNodeOpType::append) {
        TC_ASSERT(stmt->val->width() == 1);
        emit("{}_tmp->append({}({}[{}]));", snode->node_type_name,
             snode->ch[0]->node_type_name, stmt->val->raw_name(), l);
      } else if (stmt->op_type == SNodeOpType::clear) {
        emit("{}_tmp->clear();", snode->node_type_name);
      } else if (stmt->op_type == SNodeOpType::probe) {
        emit("{}[{}] = query_{}(root, {});", stmt->raw_name(), l,
             snode->node_type_name, make_list(indices, ""));
        if (snode->type == SNodeType::dynamic) {
          emit("if ({}[{}]) {{", stmt->raw_name(), l);
          emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
               snode->node_type_name, snode->node_type_name,
               make_list(indices, ""));
          emit("{}[{}] = {}_tmp->get_n();", stmt->raw_name(), l,
               snode->node_type_name);
          emit("}}");
        }
      } else if (stmt->op_type == SNodeOpType::activate) {
        emit("activate_{}(root, {});", snode->node_type_name,
             make_list(indices, ""));
      } else {
        TC_NOT_IMPLEMENTED
      }
      emit("}}");
    }
  }

  void visit(AtomicOpStmt *stmt) {
    auto mask = stmt->parent->mask();
    for (int l = 0; l < stmt->width(); l++) {
      if (mask) {
        emit("if ({}[{}]) ", mask->raw_name(), l);
      } else {
        TC_ASSERT(stmt->val->ret_type.data_type == DataType::f32 ||
                  stmt->val->ret_type.data_type == DataType::i32);
        TC_ASSERT(stmt->op_type == AtomicOpType::add);
        emit("atomic_add({}[{}], {}[{}]);", stmt->dest->raw_name(), l,
             stmt->val->raw_name(), l);
      }
    }
  }

  void visit(GlobalPtrStmt *stmt) {
    emit("{} *{}[{}];", data_type_name(stmt->ret_type.data_type),
         stmt->raw_name(), stmt->ret_type.width);
    for (int l = 0; l < stmt->ret_type.width; l++) {
      // Try to weaken here...
      std::vector<int> offsets(stmt->indices.size());

      auto snode = stmt->snodes[l];
      std::vector<std::string> indices(max_num_indices, "0");  // = "(root, ";
      for (int i = 0; i < (int)stmt->indices.size(); i++) {
        if (snode->physical_index_position[i] != -1) {
          // TC_ASSERT(snode->physical_index_position[i] != -1);
          indices[snode->physical_index_position[i]] =
              stmt->indices[i]->raw_name() + fmt::format("[{}]", l);
        }
      }
      std::string full_access = fmt::format(
          "{}[{}] = &{}_{}{}->val;", stmt->raw_name(), l,
          stmt->accessor_func_name(), stmt->snodes[l]->node_type_name,
          "(root, " + make_list(indices, "") + ")");

      bool weakened = false;
      if (current_struct_for &&
          snode->parent == current_struct_for->snode->parent) {
        bool identical_indices = false;
        bool all_offsets_zero = true;
        for (int i = 0; i < (int)stmt->indices.size(); i++) {
          auto ret = analysis::value_diff(stmt->indices[i], l,
                                          current_struct_for->loop_vars[i]);
          if (!ret.linear_related() || !ret.certain()) {
            identical_indices = false;
          }
          offsets[i] = ret.low;
          if (ret.low != 0)
            all_offsets_zero = false;
        }
        if (identical_indices) {
          // TC_WARN("Weakened addressing");
          weakened = true;

          std::string cond;
          cond = "true";
          // add safe guards...
          for (int i = 0; i < (int)stmt->indices.size(); i++) {
            if (offsets[i] == 0)
              continue;
            // TODO: fix hacky hardcoded name, make sure index same order as
            // snode indices
            std::string local_var = fmt::format(
                "index_{}_{}_local", snode->parent->node_type_name, i);
            int upper_bound = 1 << snode->parent->extractors[i].num_bits;
            if (offsets[i] == -1) {
              cond += fmt::format("&& {} > 0", local_var);
            } else if (offsets[i] >= 1) {
              cond += fmt::format("&& {} < {} - {}", local_var, upper_bound,
                                  offsets[i]);
            }
          }

          int offset = 0;
          int current_num_bits = 0;
          for (int i = (int)stmt->indices.size() - 1; i >= 0; i--) {
            offset += offsets[i] * (1 << current_num_bits);
            current_num_bits += snode->parent->extractors[i].num_bits;
          }

          emit("if ({}) {{", cond);
          emit("{}[{}] = &access_{}({}_cache, {}_loop + {})->val;",
               stmt->raw_name(), l, snode->node_type_name,
               snode->parent->node_type_name, snode->parent->node_type_name,
               offset);
          emit("}} else {{");
          emit("{}", full_access);
          emit("}}");
        }
      }
      if (!weakened) {
        emit("{}", full_access);
      }
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    if (!current_program->config.force_vectorized_global_store) {
      for (int i = 0; i < stmt->data->ret_type.width; i++) {
        if (stmt->parent->mask()) {
          TC_ASSERT(stmt->width() == 1);
          emit("if ({}[{}])", stmt->parent->mask()->raw_name(), i);
        }
        emit("*({} *){}[{}] = {}[{}];",
             data_type_name(stmt->data->ret_type.data_type),
             stmt->ptr->raw_name(), i, stmt->data->raw_name(), i);
      }
    } else {
      emit("{}.store({}[0]);", stmt->data->raw_name(), stmt->ptr->raw_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    const int width = stmt->width();
    if (get_current_program().config.attempt_vectorized_load_cpu &&
        width >= 4 && stmt->ptr->is<ElementShuffleStmt>()) {
      TC_ASSERT(stmt->ret_type.data_type == DataType::i32 ||
                stmt->ret_type.data_type == DataType::f32);

      auto shuffle = stmt->ptr->as<ElementShuffleStmt>();
      std::vector<bool> loaded(width, false);
      std::vector<Stmt *> statements(width, nullptr);
      std::vector<int> offsets(width, 0);

      for (int i = 0; i < width; i++) {
        auto src = shuffle->elements[i].stmt;
        if (shuffle->elements[i].stmt->is<IntegerOffsetStmt>()) {
          auto indir = src->as<IntegerOffsetStmt>();
          statements[i] = indir->input;
          offsets[i] = indir->offset;
        } else {
          statements[i] = src;
          offsets[i] = 0;
        }
      }

      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
      for (int i = 0; i < width; i++) {
        if (loaded[i])
          continue;
        std::vector<bool> mask(width, false);
        mask[i] = true;
        for (int j = i + 1; j < width; j++) {
          if (statements[i] == statements[j]) {
            if ((j - i) * (int)sizeof(int32) == offsets[j] - offsets[i]) {
              mask[j] = true;
            }
          }
        }
        int imm_mask = 0;
        for (int j = width - 1; j >= 0; j--) {
          if (mask[j]) {
            loaded[j] = true;
          }
          imm_mask *= 2;
          imm_mask += (int)mask[j];
        }
        // load and blend in
        if (i == 0) {
          emit("{} = {}::load({}[0]);", stmt->raw_name(),
               stmt->ret_data_type_name(),
               shuffle->elements[i].stmt->raw_name());
        } else {
          emit("{} = blend<{}>({}, {}::load({}[0] - {}));", stmt->raw_name(),
               imm_mask, stmt->raw_name(), stmt->ret_data_type_name(),
               shuffle->elements[i].stmt->raw_name(), i);
        }
      }
    } else {
      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
      for (int i = 0; i < stmt->ret_type.width; i++) {
        emit("{}[{}] = *{}[{}];", stmt->raw_name(), i, stmt->ptr->raw_name(),
             i);
      }
    }
  }

  void visit(ExternalPtrStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    TC_ASSERT(stmt->indices.size() == 1);
    auto dt = stmt->ret_type.data_type;
    emit("const {} *{}[1] = {{&{}[{}]}};", data_type_name(dt), stmt->raw_name(),
         stmt->base_ptrs[0]->raw_name(), stmt->indices[0]->raw_name());
  }

  void visit(ElementShuffleStmt *stmt) {
    auto init = stmt->elements.serialize(
        [](const VectorElement &elem) {
          return fmt::format("{}[{}]", elem.stmt->raw_name(), elem.index);
        },
        "{");
    if (stmt->pointer) {
      emit("{} * const {} [{}] {};", data_type_name(stmt->ret_type.data_type),
           stmt->raw_name(), stmt->width(), init);
    } else {
      emit("const {} {} ({});", stmt->ret_data_type_name(), stmt->raw_name(),
           init);
    }
  }

  void visit(AssertStmt *stmt) {
    emit("#if defined(TL_DEBUG)");
    emit(R"(TC_ASSERT_INFO({}, "{}");)", stmt->val->raw_name(), stmt->text);
    emit("#endif");
  }

  void visit(OffsetAndExtractBitsStmt *stmt) {
    emit(R"(auto {} = ((({} + {}) >> {}) & ((1 << {}) - 1));)",
         stmt->raw_name(), stmt->offset, stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  void visit(LinearizeStmt *stmt) {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({}) * {} + {}", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit(R"(auto {} = {};)", stmt->raw_name(), val);
  }

  void visit(IntegerOffsetStmt *stmt) {
    if (stmt->input->is<GetChStmt>() &&
        stmt->input->as<GetChStmt>()->output_snode->type == SNodeType::place) {
      auto input = stmt->input->as<GetChStmt>();
      auto dtn = input->output_snode->data_type_name();
      emit(R"({}* {}[1] {{({} *)((char *){}[0] + {})}};)", dtn,
           stmt->raw_name(), dtn, stmt->input->raw_name(), stmt->offset);
    } else {
      emit(R"(auto {} = {} + {};)", stmt->raw_name(), stmt->input->raw_name(),
           stmt->offset);
    }
  }

  void visit(SNodeLookupStmt *stmt) {
    std::string parent;
    if (stmt->input_snode) {
      parent = stmt->input_snode->raw_name();
    } else {
      parent = "root";
    }
    std::vector<std::string> global_indices(max_num_indices, "0");
    auto snode = stmt->snode;
    for (int i = 0; i < (int)stmt->global_indices.size(); i++) {
      if (snode->physical_index_position[i] != -1) {
        global_indices[snode->physical_index_position[i]] =
            stmt->global_indices[i]->raw_name() + fmt::format("[{}]", 0);
      }
    }
    if (stmt->activate && stmt->snode->type != SNodeType::place) {
      emit(R"({}->activate({}, {});)", parent, stmt->input_index->raw_name(),
           make_list(global_indices, "{"));
    }
    emit("auto {}_guarded = {}->look_up({});", stmt->raw_name(), parent,
         stmt->input_index->raw_name());
    if (!stmt->activate && snode->has_null()) {
      // safe guard with ambient node
      emit("if({}_guarded == nullptr) {}_guarded = &{}_ambient;",
           stmt->raw_name(), stmt->raw_name(), snode->node_type_name);
    }
    emit(R"(auto {} = {}_guarded;)", stmt->raw_name(), stmt->raw_name());
  }

  void visit(GetChStmt *stmt) {
    // emit("{} *{};", stmt->output_snode->data_type_name(),
    //     stmt->raw_name());
    if (stmt->output_snode->type == SNodeType::place) {
      emit(R"({} *{}[1] {{&{}->get{}()->val}};)",
           stmt->output_snode->data_type_name(), stmt->raw_name(),
           stmt->input_ptr->raw_name(), stmt->chid);
    } else {
      emit(R"(auto {} = {}->get{}();)", stmt->raw_name(),
           stmt->input_ptr->raw_name(), stmt->chid);
    }
  }
};

void CPUCodeGen::lower_cpp() {
  auto ir = kernel->ir;
  if (prog->config.print_ir) {
    TC_TRACE("Initial IR:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::lower(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Typechecked:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::slp_vectorize(ir);
  if (prog->config.print_ir) {
    TC_TRACE("SLPed:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (prog->config.print_ir) {
    TC_TRACE("LoopVeced:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::vector_split(ir, prog->config.max_vector_width,
                       prog->config.serial_schedule);
  if (prog->config.print_ir) {
    TC_TRACE("LoopSplitted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (prog->config.simplify_before_lower_access) {
    irpass::simplify(ir);
    if (prog->config.print_ir) {
      TC_TRACE("Simplified I:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (kernel->grad) {
    // irpass::re_id(ir);
    // TC_TRACE("Primal:");
    // irpass::print(ir);
    irpass::make_adjoint(ir);
    irpass::typecheck(ir);
    if (prog->config.print_ir) {
      TC_TRACE("Adjoint:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (prog->config.lower_access) {
    irpass::lower_access(ir, true);
    if (prog->config.print_ir) {
      TC_TRACE("Access Lowered:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    if (prog->config.simplify_after_lower_access) {
      irpass::die(ir);
      if (prog->config.print_ir) {
        TC_TRACE("DIEd:");
        irpass::re_id(ir);
        irpass::print(ir);
      }
      irpass::simplify(ir);
      if (prog->config.print_ir) {
        TC_TRACE("Simplified II:");
        irpass::re_id(ir);
        irpass::print(ir);
      }
    }
  }
  irpass::die(ir);
  if (prog->config.print_ir) {
    TC_TRACE("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::flag_access(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Access Flagged:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

void CPUCodeGen::lower_llvm() {
  auto ir = kernel->ir;
  if (prog->config.print_ir) {
    TC_TRACE("Initial IR:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::lower(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Typechecked:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::slp_vectorize(ir);
  if (prog->config.print_ir) {
    TC_TRACE("SLPed:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (prog->config.print_ir) {
    TC_TRACE("LoopVeced:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::vector_split(ir, prog->config.max_vector_width,
                       prog->config.serial_schedule);
  if (prog->config.print_ir) {
    TC_TRACE("LoopSplitted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (prog->config.simplify_before_lower_access) {
    irpass::simplify(ir);
    if (prog->config.print_ir) {
      TC_TRACE("Simplified I:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (kernel->grad) {
    // irpass::re_id(ir);
    // TC_TRACE("Primal:");
    // irpass::print(ir);
    irpass::make_adjoint(ir);
    irpass::typecheck(ir);
    if (prog->config.print_ir) {
      TC_TRACE("Adjoint:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  if (prog->config.lower_access) {
    irpass::lower_access(ir, true);
    if (prog->config.print_ir) {
      TC_TRACE("Access Lowered:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    if (prog->config.simplify_after_lower_access) {
      irpass::die(ir);
      if (prog->config.print_ir) {
        TC_TRACE("DIEd:");
        irpass::re_id(ir);
        irpass::print(ir);
      }
      irpass::simplify(ir);
      if (prog->config.print_ir) {
        TC_TRACE("Simplified II:");
        irpass::re_id(ir);
        irpass::print(ir);
      }
    }
  }
  irpass::die(ir);
  if (prog->config.print_ir) {
    TC_TRACE("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::flag_access(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Access Flagged:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::constant_fold(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Constant folded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::offload(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }

  irpass::full_simplify(ir);
  if (prog->config.print_ir) {
    TC_TRACE("Simplified III:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

void CPUCodeGen::lower() {
  TC_PROFILER(__FUNCTION__)
  if (prog->config.use_llvm) {
    lower_llvm();
  } else {
    lower_cpp();
  }
}

void CPUCodeGen::codegen() {
  generate_header();

  emit("extern \"C\" void " + func_name + "(Context &context) {{\n");
  emit("auto root = ({} *)context.buffers[0];",
       prog->snode_root->node_type_name);

  emit(R"(context.cpu_profiler->start("{}");)", func_name);
  CPUIRCodeGen::run(this, kernel->ir, kernel);
  emit(R"(context.cpu_profiler->stop();)", func_name);

  emit("}}\n");

  line_suffix = "";
  generate_tail();
}

TLANG_NAMESPACE_END
