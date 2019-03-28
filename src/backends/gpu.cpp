#include "gpu.h"
#include "loop_gen.h"

TLANG_NAMESPACE_BEGIN

class GPUIRCodeGen : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  GPUCodeGen *codegen;
  LoopGenerator loopgen;
  bool first_level = false;

  GPUIRCodeGen(GPUCodeGen *codegen) : codegen(codegen), loopgen(codegen) {
    current_struct_for = nullptr;
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    codegen->emit(f, std::forward<Args>(args)...);
  }

  std::string loop_variable(SNode *snode) {
    return snode->node_type_name + "_loop";
  }

  static void run(GPUCodeGen *codegen, IRNode *node) {
    auto p = GPUIRCodeGen(codegen);
    p.first_level = true;
    node->accept(&p);
  }

  void visit(Block *stmt_list) {
    if (first_level) {
      first_level = false;
      // Check structure
      // Only the last statement can be a RangeFor/StructFor
      // The rest must be Alloca for loop variables and consts for bounds
      for (int i = 0; i + 1 < stmt_list->statements.size(); i++) {
        auto s = stmt_list->statements[i].get();
        TC_ASSERT(s->is<AllocaStmt>() || s->is<ConstStmt>());
      }

      auto &for_stmt_ = stmt_list->statements.back();
      if (for_stmt_->is<RangeForStmt>()) {
        auto range_for = for_stmt_->as<RangeForStmt>();

        // GPU Kernel
        emit("__global__ void {}_kernel(Context context) {{",
             codegen->func_name);
        emit("auto root = ({} *)context.buffers[0];",
             codegen->prog->snode_root->node_type_name);
        int begin = range_for->begin->as<ConstStmt>()->val[0].val_int32();
        int end = range_for->end->as<ConstStmt>()->val[0].val_int32();

        emit("auto {} = blockIdx.x * blockDim.x + threadIdx.x + {};\n",
             range_for->loop_var->raw_name(), begin);
        emit("if ({} >= {}) return;", range_for->loop_var->raw_name(), end);

        range_for->body->accept(this);

        // CPU Kernel code
        emit("}}\n\n");
        emit("extern \"C\" void {} (Context context) {{\n", codegen->func_name);

        TC_ASSERT(begin == 0);

        int block_size = 256;
        int num_blocks = (end - begin + block_size - 1) / block_size;
        emit("{}_kernel<<<{}, {}>>>(context);", codegen->func_name, num_blocks,
             block_size);
      } else {
        // struct for
        TC_ASSERT_INFO(current_struct_for == nullptr,
                       "Struct for cannot be nested.");
        auto for_stmt = for_stmt_->as<StructForStmt>();
        current_struct_for = for_stmt;
        auto leaf = for_stmt->snode->parent;

        emit("__global__ void {}_kernel(Context context) {{",
             codegen->func_name);
        emit("auto root = ({} *)context.buffers[0];",
             codegen->prog->snode_root->node_type_name);

        emit("auto leaves = (LeafContext<{}> *)(context.leaves);",
             codegen->prog->snode_root->node_type_name);
        emit("auto num_leaves = context.num_leaves;",
             codegen->prog->snode_root->node_type_name);
        emit("auto leaf_loop = blockIdx.x;",
             codegen->prog->snode_root->node_type_name);
        emit("if (leaf_loop >= num_leaves) return;");

        loopgen.emit_load_from_context(leaf);
        emit("auto {} = threadIdx.x;", loopgen.loop_variable(leaf));
        loopgen.update_indices(leaf);
        loopgen.emit_setup_loop_variables(for_stmt, leaf);
        for_stmt->body->accept(this);

        emit("}}");

        emit("extern \"C\" void {} (Context context) {{\n", codegen->func_name);
        emit("auto root = ({} *)context.buffers[0];",
             current_program->snode_root->node_type_name);
        emit("{{");

        loopgen.loop_gen_leaves(for_stmt, leaf);

        std::string vars;
        for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
          vars += for_stmt->loop_vars[i]->raw_name();
          if (i + 1 < for_stmt->loop_vars.size()) {
            vars += ",";
          }
        }
        emit("context.num_leaves = leaves.size();");
        emit("auto list_size = sizeof(LeafContext<{}>) * context.num_leaves;",
             leaf->node_type_name);
        emit("cudaMalloc(&context.leaves, list_size);");
        emit(
            "cudaMemcpy(context.leaves, leaves.data(), list_size, "
            "cudaMemcpyHostToDevice);");
        emit("printf(\"num leaves %d\\n\", context.num_leaves);");
        // allocate the vector...

        emit("{}_kernel<<<context.num_leaves, {}().get_n()>>>(context);",
             codegen->func_name, leaf->node_type_name);

        emit("cudaFree(context.leaves); context.leaves = nullptr;");

        emit("}}");
        current_struct_for = nullptr;
      }

      emit("cudaDeviceSynchronize();\n");
      emit("auto err = cudaGetLastError();");
      emit("if (err) {{");
      emit(
          "printf(\"CUDA Error (File %s Ln %d): %s\\n\", "
          "__FILE__, __LINE__, cudaGetErrorString(err));");
      emit("exit(-1);}}");
      emit("}}\n");
    } else {
      for (auto &stmt : stmt_list->statements) {
        stmt->accept(this);
      }
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
      emit("const {} {}(static_cast<{}>({}));", stmt->ret_data_type_name(),
           stmt->raw_name(), data_type_name(stmt->cast_type),
           stmt->rhs->raw_name());
    }
  }

  void visit(BinaryOpStmt *bin) {
    emit("const {} {}({}({}, {}));", bin->ret_data_type_name(), bin->raw_name(),
         binary_type_name(bin->op_type), bin->lhs->raw_name(),
         bin->rhs->raw_name());
  }

  void visit(TrinaryOpStmt *tri) {
    TC_ASSERT(tri->op_type == TrinaryType::select);
    emit("const {} {} = {} ? {} : {};", tri->ret_data_type_name(),
         tri->raw_name(), tri->op1->raw_name(), tri->op2->raw_name(),
         tri->op3->raw_name());
  }

  void visit(AtomicOpStmt *stmt) {
    TC_ASSERT(stmt->val->ret_type.data_type == DataType::f32 ||
              stmt->val->ret_type.data_type == DataType::i32);
    TC_ASSERT(stmt->op_type == AtomicType::add);
    emit("atomicAdd({}[0], {});", stmt->dest->raw_name(),
         stmt->val->raw_name());
  }

  void visit(IfStmt *if_stmt) {
    emit("if ({}) {{", if_stmt->cond->raw_name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      emit("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(PrintStmt *print_stmt) {
    if (print_stmt->stmt->ret_type.data_type == DataType::i32) {
      emit("printf(\"{}\" \" = %d\\n\", {});", print_stmt->str,
           print_stmt->stmt->raw_name());
    } else if (print_stmt->stmt->ret_type.data_type == DataType::f32) {
      emit("printf(\"{}\" \" = %f\\n\", {});", print_stmt->str,
           print_stmt->stmt->raw_name());
    } else {
      TC_NOT_IMPLEMENTED
    }
  }

  void visit(ConstStmt *const_stmt) {
    emit("const {} {}({});", const_stmt->ret_type.str(), const_stmt->raw_name(),
         const_stmt->val.serialize(
             [&](const TypedConstant &t) { return t.stringify(); }, "{"));
  }

  void visit(WhileControlStmt *stmt) {
    emit("if (!{}) break;", stmt->mask->raw_name());
  }

  void visit(WhileStmt *stmt) {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(StructForStmt *for_stmt) {
    // generate_loop_header(for_stmt->snode, for_stmt, true);
    TC_ASSERT_INFO(current_struct_for == nullptr,
                   "Structu for cannot be nested.");
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
    // generate_loop_tail(for_stmt->snode, for_stmt, true);
  }

  void visit(RangeForStmt *for_stmt) {
    auto loop_var = for_stmt->loop_var;
    if (for_stmt->parallelize) {
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
      // Try to weaken here...
      std::vector<int> offsets(stmt->indices.size());

      std::string indices = "(root, ";
      for (int i = 0; i < max_num_indices; i++) {
        if (i < (int)stmt->indices.size()) {
          indices += stmt->indices[i]->raw_name();
        } else {
          indices += "0";
        }
        if (i + 1 < max_num_indices)
          indices += ",";
      }
      indices += ")";
      std::string strong_access =
          fmt::format("{}[{}] = access_{}{};", stmt->raw_name(), l,
                      stmt->snode[l]->node_type_name, indices);

      bool weakened = false;
      auto snode = stmt->snode[l];
      if (current_struct_for &&
          snode->parent == current_struct_for->snode->parent) {
        bool identical_indices = true;
        bool all_offsets_zero = true;
        for (int i = 0; i < stmt->indices.size(); i++) {
          auto ret = analysis::value_diff(stmt->indices[i], l,
                                          current_struct_for->loop_vars[i]);
          if (!ret.first) {
            identical_indices = false;
          }
          offsets[i] = ret.second;
          if (ret.second != 0)
            all_offsets_zero = false;
        }
        if (false && identical_indices) {
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
            } else if (offsets[i] == 1) {
              cond += fmt::format("&& {} < {} - 1", local_var, upper_bound);
            } else {
              TC_NOT_IMPLEMENTED;
            }
          }

          TC_WARN("offset can be wrong in multidimensional cases");
          int offset = offsets[0];
          emit("if ({}) {{", cond);
          emit("{}[{}] = access_{}({}_cache, {}_loop + {});", stmt->raw_name(),
               l, snode->node_type_name, snode->parent->node_type_name,
               snode->parent->node_type_name, offset);
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
      emit("*({} *){}[{}] = {};",
           data_type_name(stmt->data->ret_type.data_type),
           stmt->ptr->raw_name(), 0, stmt->data->raw_name());
    } else {
      emit("{}.store({});", stmt->data->raw_name(), stmt->ptr->raw_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    emit("const auto {} = *({}[0]);", stmt->raw_name(), stmt->ptr->raw_name());
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

void GPUCodeGen::lower() {
  auto ir = kernel->ir;
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
  irpass::eliminate_dup(ir);
  if (prog->config.print_ir)
    irpass::print(ir);
}

void GPUCodeGen::codegen() {
  emit("#define TC_GPU");
  generate_header();

  // Body
  GPUIRCodeGen::run(this, kernel->ir);

  line_suffix = "";
  generate_tail();
}

TLANG_NAMESPACE_END

#if (0)
#endif
