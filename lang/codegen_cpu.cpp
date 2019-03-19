#include <taichi/common/util.h>
#include <taichi/io/io.h>
#include <set>

#include "util.h"
#include "codegen_cpu.h"
#include "program.h"
#include "ir.h"

TLANG_NAMESPACE_BEGIN

class IRCodeGen : public IRVisitor {
 public:
  CodeGenBase *codegen;
  IRCodeGen(CodeGenBase *codegen) : codegen(codegen) {
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    codegen->emit(f, std::forward<Args>(args)...);
  }

  std::string loop_variable(SNode *snode) {
    return snode->node_type_name + "_loop";
  }

  std::string index_name_local(SNode *snode, int i) {
    return fmt::format("index_{}_{}_local", snode->node_type_name, i);
  }

  std::string index_name_global(SNode *snode, int i) {
    return fmt::format("index_{}_{}_global", snode->node_type_name, i);
  }

#define CODE_REGION_VAR(x)
#define CODE_REGION(x)
#define emit_code emit

  void generate_loop_header(SNode *snode,
                            StructuralForStmt *stmt,
                            bool last_level = false) {
    if (snode->parent != nullptr) {
      generate_loop_header(snode->parent, stmt,
                           last_level && snode->type == SNodeType::forked);
    } else {
      return;  // no loop for root, which is a fork
    }
    auto l = loop_variable(snode);
    bool interior = last_level && snode->type != SNodeType::forked;
    /*
    CodeRegion r;
    if (last_level)
      r = CodeRegion::interior_loop_begin;
    else
      r = CodeRegion::exterior_loop_begin;
    */
    CODE_REGION_VAR(r);
    if (snode->parent->parent == nullptr)
      emit_code("auto {} = 0;", loop_variable(snode->parent));
    auto parent = fmt::format("{}_cache", snode->parent->node_type_name);
    emit_code("auto {}_cache = access_{}({}, {});", snode->node_type_name,
              snode->node_type_name, parent, loop_variable(snode->parent));
    emit_code("int {};", l);

    if (snode->type == SNodeType::pointer) {
      emit_code("if (!{}_cache->data) continue;", snode->node_type_name, l);
    }

    if (snode->type != SNodeType::hashed) {
      emit_code("auto {}_cache_n = {}_cache->get_n();", snode->node_type_name,
                snode->node_type_name);
    }
    if (snode->_multi_threaded) {
      auto p = snode->parent;
      while (p) {
        TC_ASSERT(!p->_multi_threaded);
        p = p->parent;
      }
      emit_code("#pragma omp parallel for");
    }
    // TODO: replace with vectorize width
    int parallel_instances = 1;
    auto has_residual = false;
    if (interior) {
      if (!has_residual) {
        emit_code("for ({} = 0; {} < {}_cache_n; {} += {}) {{", l, l,
                  snode->node_type_name, l, parallel_instances);
      } else {
        int residual =
            parallel_instances > 1  // when only one instance, no residual loop.
                ? 0
                : parallel_instances;
        emit_code("for ({} = 0; {} + {} < {}_cache_n; {} += {}) {{", l, l,
                  residual, snode->node_type_name, l, parallel_instances

        );
      }
    } else {
      if (snode->type == SNodeType::hashed) {
        emit_code("for (auto &{}_it : {}_cache->data) {{", l,
                  snode->node_type_name);
        emit_code("int {} = {}_it.first;", l, l);
      } else {
        emit_code("for ({} = 0; {} < {}_cache_n; {} += {}) {{", l, l,
                  snode->node_type_name, l, 1);
      }
    }

    if (has_residual && last_level) {
      CODE_REGION(residual_begin);  // TODO: DRY..
      emit_code("if ({} < {}_cache_n) {{", l, snode->node_type_name);
    }
    // update indices....
    for (int i = 0; i < max_num_indices; i++) {
      std::string ancester = "0 |";
      if (snode->parent->parent != nullptr) {
        ancester = index_name_global(snode->parent, i) + " |";
      }
      std::string addition = "0";
      if (snode->extractors[i].num_bits) {
        addition = fmt::format(
            "((({} >> {}) & ((1 << {}) - 1)) << {})", l,
            snode->extractors[i].dest_offset - snode->total_bit_start,
            snode->extractors[i].num_bits, snode->extractors[i].start);
      }
      emit_code("int {} = {};", index_name_local(snode, i), addition);
      emit_code("int {} = {} {};", index_name_global(snode, i), ancester,
                index_name_local(snode, i));
      if (has_residual && last_level) {
        CODE_REGION(residual_begin);  // TODO: DRY..
        emit_code("int {} = {};", index_name_local(snode, i), addition);
        emit_code("int {} = {} {};", index_name_global(snode, i), ancester,
                  index_name_local(snode, i));
      }
    }
    if (has_residual && last_level) {
      CODE_REGION(residual_end);
      emit_code("}}");
    }
  }

  void generate_loop_tail(SNode *snode,
                          StructuralForStmt *stmt,
                          bool last_level = false) {
    /*
    CodeRegion r;
    r = CodeRegion::exterior_loop_end;
    auto l = loop_variable(snode);
    if (last_level && snode->type != SNodeType::forked) {
      // emit_code("{} += {}; b += {};", l, num_groups * unroll, unroll);
      r = CodeRegion::interior_loop_end;
    }
    */
    CODE_REGION_VAR(r);
    if (snode->parent != nullptr) {
      CODE_REGION_VAR(last_level ? CodeRegion::interior_loop_end
                                 : CodeRegion::exterior_loop_end);
      emit_code("}}\n");
      generate_loop_tail(snode->parent, stmt,
                         last_level && snode->type == SNodeType::forked);
    } else {
      return;  // no loop for root, which is a fork
    }
  }

#undef emit_code

  static void run(CodeGenBase *codegen, IRNode *node) {
    auto p = IRCodeGen(codegen);
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

  void visit(BinaryOpStmt *bin) {
    emit("const {} {}({}({}, {}));", bin->ret_data_type_name(), bin->raw_name(),
         binary_type_name(bin->op_type), bin->lhs->raw_name(),
         bin->rhs->raw_name());
  }

  void visit(UnaryOpStmt *stmt) {
    if (stmt->op_type != UnaryType::cast) {
      emit("const {} {}({}({}));", stmt->ret_data_type_name(), stmt->raw_name(),
           unary_type_name(stmt->op_type), stmt->rhs->raw_name());
    } else {
      emit("const {} {}(cast<{}>({}));", stmt->ret_data_type_name(),
           stmt->raw_name(), data_type_name(stmt->cast_type),
           stmt->rhs->raw_name());
    }
  }

  void visit(IfStmt *if_stmt) {
    // emit("if ({}) {{", if_stmt->cond->raw_name());
    emit("{{");
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      // emit("}} else {{");
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

  void visit(StructuralForStmt *for_stmt) {
    auto loop_var = for_stmt->loop_var;
    generate_loop_header(for_stmt->snode, for_stmt);
    for_stmt->body->accept(this);
    generate_loop_tail(for_stmt->snode, for_stmt);
    emit("}}");
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

  void visit(GlobalPtrStmt *stmt) {
    emit("{} *{}[{}];", data_type_name(stmt->ret_type.data_type),
         stmt->raw_name(), stmt->ret_type.width);
    for (int l = 0; l < stmt->ret_type.width; l++) {
      std::string indices = "(root, ";
      for (int i = 0; i < max_num_indices; i++) {
        if (i < (int)stmt->indices.size()) {
          indices += stmt->indices[i]->raw_name() + fmt::format("[{}]", l);
        } else {
          indices += "0";
        }
        if (i + 1 < max_num_indices)
          indices += ",";
      }
      indices += ")";
      emit("{}[{}] = access_{}{};", stmt->raw_name(), l,
           stmt->snode[l]->node_type_name, indices);
    }
  }

  void visit(GlobalStoreStmt *stmt) {
    if (!current_program->config.force_vectorized_global_store) {
      for (int i = 0; i < stmt->data->ret_type.width; i++) {
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
};

void CPUCodeGen::codegen(Kernel &kernel) {
  // TC_ASSERT(mode == Mode::vector);
  this->prog = &kernel.program;
  this->current_kernel = &kernel;

  {
    CODE_REGION(header);
    generate_header();
  }

  auto ir = kernel.ir;
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::lower(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::slp_vectorize(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::loop_vectorize(ir);
  if (prog->config.print_ir)
    irpass::print(ir);
  irpass::vector_split(ir, prog->config.max_vector_width,
                       prog->config.serial_schedule);
  if (prog->config.print_ir)
    irpass::print(ir);
  irpass::eliminate_dup(ir);
  if (prog->config.print_ir)
    irpass::print(ir);
  IRCodeGen::run(this, ir);

  {
    CODE_REGION(tail);
    code_suffix = "";
    generate_tail();
  }
}

FunctionType CPUCodeGen::get(Program &prog, Kernel &kernel) {
  // auto mode = CPUCodeGen::Mode::vv;
  auto mode = CPUCodeGen::Mode::intrinsics;
  auto simd_width = prog.config.simd_width;
  this->mode = mode;
  this->simd_width = simd_width;
  codegen(kernel);
  return compile();
}

FunctionType Program::compile(Kernel &kernel) {
  FunctionType ret = nullptr;
  if (config.arch == Arch::x86_64) {
    CPUCodeGen codegen;
    if (!kernel.name.empty()) {
      codegen.source_name = kernel.name + ".cpp";
    }
    ret = codegen.get(*this, kernel);
  } else if (config.arch == Arch::gpu) {
    TC_NOT_IMPLEMENTED
    // GPUCodeGen codegen;
    // function = codegen.get(*this);
  } else {
    TC_NOT_IMPLEMENTED;
  }
  TC_ASSERT(ret);
  return ret;
}

std::string CodeGenBase::get_source_fn() {
  return fmt::format("{}/{}/{}", get_project_fn(), folder, source_name);
}

FunctionType CPUCodeGen::compile() {
  write_code_to_file();
  auto cmd = get_current_program().config.compile_cmd(get_source_fn(),
                                                      get_library_fn());
  auto compile_ret = std::system(cmd.c_str());
  if (compile_ret != 0) {
    auto cmd = get_current_program().config.compile_cmd(get_source_fn(),
                                                        get_library_fn(), true);
    trash(std::system(cmd.c_str()));
    TC_ERROR("Source {} compilation failed.", get_source_fn());
  }
  disassemble();
  return load_function();
}

TLANG_NAMESPACE_END
