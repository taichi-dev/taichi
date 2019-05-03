#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "../util.h"
#include "cpu.h"
#include "loop_gen.h"
#include "../program.h"
#include "../ir.h"

TLANG_NAMESPACE_BEGIN

class CPUIRCodeGen : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  CodeGenBase *codegen;
  LoopGenerator loopgen;

  CPUIRCodeGen(CodeGenBase *codegen) : codegen(codegen), loopgen(codegen) {
    current_struct_for = nullptr;
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    codegen->emit(f, std::forward<Args>(args)...);
  }

  static void run(CodeGenBase *codegen, IRNode *node) {
    auto p = CPUIRCodeGen(codegen);
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
    if (stmt->op_type != UnaryType::cast) {
      emit("const {} {}({}({}));", stmt->ret_data_type_name(), stmt->raw_name(),
           unary_type_name(stmt->op_type), stmt->rhs->raw_name());
    } else {
      if (stmt->cast_by_value) {
        emit("const {} {}(cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->rhs->raw_name());
      } else {
        emit("const {} {}(union_cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->rhs->raw_name());
      }
    }
  }

  void visit(BinaryOpStmt *bin) {
    emit("const {} {}({}({}, {}));", bin->ret_data_type_name(), bin->raw_name(),
         binary_type_name(bin->op_type), bin->lhs->raw_name(),
         bin->rhs->raw_name());
  }

  void visit(TrinaryOpStmt *tri) {
    emit("const {} {}({}({}, {}, {}));", tri->ret_data_type_name(),
         tri->raw_name(), trinary_type_name(tri->op_type), tri->op1->raw_name(),
         tri->op2->raw_name(), tri->op3->raw_name());
  }

  void visit(IfStmt *if_stmt) {
    emit("{{");
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      emit("}}  {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(PrintStmt *print_stmt) {
    emit("std::cout << \"[debug] \" \"{}\" \" = \" << {} << std::endl;",
         print_stmt->str, print_stmt->stmt->raw_name());
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
    for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
      vars += for_stmt->loop_vars[i]->raw_name();
      if (i + 1 < for_stmt->loop_vars.size()) {
        vars += ",";
      }
    }
    emit("int num_leaves = leaves.size();");
    if (for_stmt->parallelize) {
      emit("omp_set_num_threads({});", for_stmt->parallelize);
      emit("#pragma omp parallel for private({})", vars);
    }
    emit("for (int leaf_loop = 0; leaf_loop < num_leaves; leaf_loop++) {{");
    loopgen.emit_load_from_context(leaf);
    loopgen.generate_single_loop_header(leaf, true, for_stmt->vectorize);
    loopgen.emit_setup_loop_variables(for_stmt, leaf);
    for_stmt->body->accept(this);
    emit("}}");
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
      emit("#pragma omp parallel for private({})", loop_var->raw_name());
    }
    if (loop_var->ret_type.width == 1 &&
        loop_var->ret_type.data_type == DataType::i32) {
      emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
           loop_var->raw_name(), for_stmt->begin->raw_name(),
           loop_var->raw_name(), for_stmt->end->raw_name(),
           loop_var->raw_name(), loop_var->raw_name(), for_stmt->vectorize);
      emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
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
    TC_ASSERT(stmt->width() == 1);
    auto snode = stmt->snodes[0];
    auto indices = indices_str(snode, 0, stmt->indices);

    if (stmt->op_type == SNodeOpType::probe) {
      emit("int32x1 {};", stmt->raw_name());
    }

    emit("{{");
    if (stmt->op_type != SNodeOpType::activate)
      emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
           snode->node_type_name, snode->node_type_name,
           make_list(indices, ""));
    if (stmt->op_type == SNodeOpType::append) {
      TC_ASSERT(stmt->val->width() == 1);
      emit("{}_tmp->append({}({}[0]));", snode->node_type_name,
           snode->ch[0]->node_type_name, stmt->val->raw_name());
    } else if (stmt->op_type == SNodeOpType::clear) {
      emit("{}_tmp->clear();", snode->node_type_name);
    } else if (stmt->op_type == SNodeOpType::probe) {
      emit("{}[0] = {}_tmp->get_n();", stmt->raw_name(), snode->node_type_name);
    } else if (stmt->op_type == SNodeOpType::activate) {
      emit("activate_{}(root, {});", snode->node_type_name,
           make_list(indices, ""));
    } else {
      TC_NOT_IMPLEMENTED
    }
    emit("}}");
  }

  void visit(AtomicOpStmt *stmt) {
    auto mask = stmt->parent->mask();
    for (int l = 0; l < stmt->width(); l++) {
      if (mask) {
        emit("if ({}[{}]) ", mask->raw_name(), l);
      } else {
        TC_ASSERT(stmt->val->ret_type.data_type == DataType::f32 ||
                  stmt->val->ret_type.data_type == DataType::i32);
        TC_ASSERT(stmt->op_type == AtomicType::add);
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
      for (int i = 0; i < stmt->indices.size(); i++) {
        if (snode->physical_index_position[i] != -1) {
          // TC_ASSERT(snode->physical_index_position[i] != -1);
          indices[snode->physical_index_position[i]] =
              stmt->indices[i]->raw_name() + fmt::format("[{}]", l);
        }
      }
      std::string strong_access =
          fmt::format("{}[{}] = &access_{}{}->val;", stmt->raw_name(), l,
                      stmt->snodes[l]->node_type_name,
                      "(root, " + make_list(indices, "") + ")");

      bool weakened = false;
      if (current_struct_for &&
          snode->parent == current_struct_for->snode->parent) {
        bool identical_indices = true;
        bool all_offsets_zero = true;
        for (int i = 0; i < stmt->indices.size(); i++) {
          auto ret = analysis::value_diff(stmt->indices[i], l,
                                          current_struct_for->loop_vars[i]);
          if (!ret.related || !ret.certain()) {
            identical_indices = false;
          }
          offsets[i] = ret.low;
          if (ret.low != 0)
            all_offsets_zero = false;
        }
        if (identical_indices) {
          TC_WARN("Weakened addressing");
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
          emit("{}", strong_access);
          emit("}}");
        }
      }
      if (!weakened) {
        emit("{}", strong_access);
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
    if (!current_program->config.force_vectorized_global_load) {
      emit("{} {};", stmt->ret_data_type_name(), stmt->raw_name());
      for (int i = 0; i < stmt->ret_type.width; i++) {
        emit("{}[{}] = *{}[{}];", stmt->raw_name(), i, stmt->ptr->raw_name(),
             i);
      }
    } else {
      emit("const auto {} = {}::load({}[0]);", stmt->raw_name(),
           stmt->ret_data_type_name(), stmt->ptr->raw_name());
    }
  }

  void visit(ElementShuffleStmt *stmt) {
    emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
         stmt->elements.serialize(
             [](const VectorElement &elem) {
               return fmt::format("{}[{}]", elem.stmt->raw_name(), elem.index);
             },
             "{"));
  }

  void visit(AssertStmt *stmt) {
    emit("#if defined(TL_DEBUG)");
    emit(R"(TC_ASSERT_INFO({}, "{}");)", stmt->val->raw_name(), stmt->text);
    emit("#endif");
  }
};

void CPUCodeGen::lower() {
  auto ir = kernel->ir;
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::lower(ir);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::slp_vectorize(ir);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::vector_split(ir, prog->config.max_vector_width,
                       prog->config.serial_schedule);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  // irpass::initialize_scratch_pad(ir);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::eliminate_dup(ir);
  if (prog->config.print_ir) {
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

void CPUCodeGen::codegen() {
  generate_header();

  emit("extern \"C\" void " + func_name + "(Context context) {{\n");
  emit("auto root = ({} *)context.buffers[0];",
       prog->snode_root->node_type_name);

  emit(R"(context.cpu_profiler->start("{}");)", func_name);
  CPUIRCodeGen::run(this, kernel->ir);
  emit(R"(context.cpu_profiler->stop();)", func_name);

  emit("}}\n");

  line_suffix = "";
  generate_tail();
}

TLANG_NAMESPACE_END
