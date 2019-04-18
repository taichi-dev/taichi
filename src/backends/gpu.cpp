#include "gpu.h"
#include "loop_gen.h"
#include "../scratch_pad.h"

TLANG_NAMESPACE_BEGIN

class GPUIRCodeGen : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  ScratchPads *current_scratch_pads;
  GPUCodeGen *codegen;
  LoopGenerator loopgen;
  bool first_level = false;

  GPUIRCodeGen(GPUCodeGen *codegen) : codegen(codegen), loopgen(codegen) {
    current_struct_for = nullptr;
    current_scratch_pads = nullptr;
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

  void struct_for_old(Stmt *for_stmt_) {
    // struct for
    TC_ASSERT_INFO(current_struct_for == nullptr,
                   "Struct for cannot be nested.");
    auto for_stmt = for_stmt_->as<StructForStmt>();
    current_struct_for = for_stmt;
    auto leaf = for_stmt->snode->parent;

    int block_division = 1;
    if (for_stmt->block_size != 0) {
      TC_ASSERT((1 << leaf->total_num_bits) % for_stmt->block_size == 0);
      block_division = (1 << leaf->total_num_bits) / for_stmt->block_size;
    }

    emit("__global__ void {}_kernel(Context context) {{", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         codegen->prog->snode_root->node_type_name);

    emit("auto leaves = (LeafContext<{}> *)(context.leaves);",
         leaf->node_type_name);
    emit("auto num_leaves = context.num_leaves;");
    emit("auto leaf_loop = blockIdx.x / {};", block_division);
    emit("if (leaf_loop >= num_leaves) return;");

    loopgen.emit_load_from_context(leaf);
    emit("auto {} = {} * (blockIdx.x % {}) + threadIdx.x;",
         loopgen.loop_variable(leaf),
         (1 << leaf->total_num_bits) / block_division, block_division);

    loopgen.update_indices(leaf);
    loopgen.emit_setup_loop_variables(for_stmt, leaf);

    std::unique_ptr<ScratchPads> scratch_pads;

    auto access_global = [&](SNode *snode) -> std::string {
      std::vector<std::string> indices(max_num_indices, "0");
      for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
        if (snode->physical_index_position[i] != -1) {
          auto var = for_stmt->loop_vars[i]->raw_name();
          indices[snode->physical_index_position[i]] =
              var + "_base " + " + " +
              (*scratch_pads).pads[snode].extract_offset("flat_index", i);
        }
      }
      return fmt::format("access_{}{}", snode->node_type_name,
                         "(root, " + make_list(indices, "") + ")");
    };

    auto for_every_element = [&](SNode *snode, ScratchPad &pad,
                                 std::string statement) {
      emit("{{");
      emit("int flat_index = threadIdx.x;");
      emit("while (flat_index < {}) {{", pad.linear_size());
      emit(statement);
      emit("flat_index += blockDim.x;");
      emit("}}");
      emit("}}");
    };

    if (!for_stmt->scratch_opt.empty()) {
      scratch_pads = irpass::initialize_scratch_pad(for_stmt);
      for (auto &pad : scratch_pads->pads) {
        emit("__shared__ {}::val_type {}[{}];", pad.first->node_type_name,
             pad.second.name(), pad.second.linear_size());
        TC_ASSERT(pad.second.is_pure());
        if (pad.second.total_flags == AccessFlag::read ||
            pad.second.total_flags == AccessFlag::accumulate) {
          // read & accumulate case
          // load from global if read
          std::string source = pad.second.total_flags == AccessFlag::read
                                   ? "*" + access_global(pad.first)
                                   : "0";
          auto statement =
              fmt::format("{}[flat_index] = {};", pad.second.name(), source);
          for_every_element(pad.first, pad.second, statement);
        }
      }
      emit("__syncthreads();");
    }

    if (leaf->type == SNodeType::dynamic) {
      emit("if ({} < {}_cache->get_n())", loopgen.loop_variable(leaf),
           leaf->node_type_name);
    }

    emit("{{");

    current_scratch_pads = scratch_pads.get();
    for_stmt->body->accept(this);
    current_scratch_pads = nullptr;

    emit("}}");

    bool needs_write_back = false;
    if (!for_stmt->scratch_opt.empty()) {
      for (auto &pad : scratch_pads->pads) {
        if (pad.second.total_flags == AccessFlag::write ||
            pad.second.total_flags == AccessFlag::accumulate) {
          needs_write_back = true;
        }
      }
    }

    if (needs_write_back) {
      emit("__syncthreads();");
      for (auto &pad : scratch_pads->pads) {
        if (pad.second.total_flags == AccessFlag::write ||
            pad.second.total_flags == AccessFlag::accumulate) {
          std::string source = access_global(pad.first);
          std::string statement;
          if (pad.second.total_flags == AccessFlag::accumulate)
            statement = fmt::format(
                "if ({}[flat_index] != 0) atomicAdd(&{}->val, "
                "{}[flat_index]);",
                pad.second.name(), source, pad.second.name());
          else
            statement =
                fmt::format("*{} = {}[flat_index];", source, pad.second.name());
          for_every_element(pad.first, pad.second, statement);
        }
      }
    }

    emit("}}");

    emit("extern \"C\" void {} (Context context) {{\n", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         current_program->snode_root->node_type_name);
    emit("{{");

    emit("auto t = get_time();");
    loopgen.loop_gen_leaves(for_stmt, leaf);

    std::string vars;
    for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
      vars += for_stmt->loop_vars[i]->raw_name();
      if (i + 1 < for_stmt->loop_vars.size()) {
        vars += ",";
      }
    }
    emit("gpu_runtime_init();");

    emit("context.num_leaves = leaves.size();");
    emit("auto list_size = sizeof(LeafContext<{}>) * context.num_leaves;",
         leaf->node_type_name);
    emit("cudaMalloc(&context.leaves, list_size);");
    emit(
        "cudaMemcpy(context.leaves, leaves.data(), list_size, "
        "cudaMemcpyHostToDevice);");
    // allocate the vector...

    emit(
        "int gridDim = context.num_leaves * {}, blockDim = ({}::get_max_n()"
        "+ {} - 1) / {};",
        block_division, leaf->node_type_name, block_division, block_division);
    emit(
        "std::cout << \"list    \" << (get_time() - t) * 1000 << \" ms\" << "
        "std::endl;");
    emit(
        "printf(\"launching kernel {} <<<%d, %d>>> num_leaves = %d\\n\", "
        "gridDim, blockDim, context.num_leaves);",
        codegen->func_name);
    emit("cudaEvent_t start, stop;");

    if (current_program->get_current_kernel().benchmarking) {
      emit("while(1) {{");
    }

    emit("cudaEventCreate(&start);");
    emit("cudaEventCreate(&stop);");
    emit("cudaEventRecord(start);");
    emit("{}_kernel<<<gridDim, blockDim>>>(context);", codegen->func_name);
    emit("cudaEventRecord(stop);");

    // emit("t = get_time();");
    // emit("std::cout << (get_time() - t) * 1000 << std::endl;");
    emit("cudaEventSynchronize(stop);");

    emit("float milliseconds = 0;");
    emit("cudaEventElapsedTime(&milliseconds, start, stop);");
    emit("std::cout << \"device  \" << milliseconds << \" ms\" << std::endl;");

    if (current_program->current_kernel->benchmarking) {
      emit("cudaDeviceSynchronize();\n");
      emit("auto err = cudaGetLastError();");
      emit("if (err) {{");
      emit(
          "printf(\"CUDA Error (File %s Ln %d): %s\\n\", "
          "__FILE__, __LINE__, cudaGetErrorString(err));");
      emit("exit(-1);}}");
      emit("}}");
    }

    emit("cudaFree(context.leaves); context.leaves = nullptr;");
    emit("}}");
    current_struct_for = nullptr;
  }

  void struct_for_new(Stmt *for_stmt_) {
    // struct for
    TC_ASSERT_INFO(current_struct_for == nullptr,
                   "Struct for cannot be nested.");
    auto for_stmt = for_stmt_->as<StructForStmt>();
    current_struct_for = for_stmt;
    auto leaf = for_stmt->snode->parent;

    int block_division = 1;
    if (for_stmt->block_size != 0) {
      TC_ASSERT((1 << leaf->total_num_bits) % for_stmt->block_size == 0);
      block_division = (1 << leaf->total_num_bits) / for_stmt->block_size;
    }

    emit("__global__ void {}_kernel(Context context) {{", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         codegen->prog->snode_root->node_type_name);

    emit("auto leaves = (SNodeMeta *)(context.leaves);",
         leaf->parent->node_type_name);
    emit("auto num_leaves = context.num_leaves;");
    emit("auto leaf_loop = blockIdx.x / {};", block_division);
    emit("if (leaf_loop >= num_leaves) return;");

    emit("auto {}_cache = ({} *)leaves[leaf_loop].ptr;", leaf->node_type_name,
         leaf->node_type_name);
    for (int i = 0; i < max_num_indices; i++) {
      emit("auto {} = leaves[leaf_loop].indices[{}];",
           loopgen.index_name_global(leaf->parent, i), i);
    }
    emit("auto {} = {} * (blockIdx.x % {}) + threadIdx.x;",
         loopgen.loop_variable(leaf),
         (1 << leaf->total_num_bits) / block_division, block_division);

    loopgen.update_indices(leaf);
    loopgen.emit_setup_loop_variables(for_stmt, leaf);

    std::unique_ptr<ScratchPads> scratch_pads;

    auto access_global = [&](SNode *snode) -> std::string {
      std::vector<std::string> indices(max_num_indices, "0");
      for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
        if (snode->physical_index_position[i] != -1) {
          auto var = for_stmt->loop_vars[i]->raw_name();
          indices[snode->physical_index_position[i]] =
              var + "_base " + " + " +
              (*scratch_pads).pads[snode].extract_offset("flat_index", i);
        }
      }
      return fmt::format("access_{}{}", snode->node_type_name,
                         "(root, " + make_list(indices, "") + ")");
    };

    auto for_every_element = [&](SNode *snode, ScratchPad &pad,
                                 std::string statement) {
      emit("{{");
      emit("int flat_index = threadIdx.x;");
      emit("while (flat_index < {}) {{", pad.linear_size());
      emit(statement);
      emit("flat_index += blockDim.x;");
      emit("}}");
      emit("}}");
    };

    if (!for_stmt->scratch_opt.empty()) {
      scratch_pads = irpass::initialize_scratch_pad(for_stmt);
      for (auto &pad : scratch_pads->pads) {
        emit("__shared__ {}::val_type {}[{}];", pad.first->node_type_name,
             pad.second.name(), pad.second.linear_size());
        TC_ASSERT(pad.second.is_pure());
        if (pad.second.total_flags == AccessFlag::read ||
            pad.second.total_flags == AccessFlag::accumulate) {
          // read & accumulate case
          // load from global if read
          std::string source = pad.second.total_flags == AccessFlag::read
                                   ? "*" + access_global(pad.first)
                                   : "0";
          auto statement =
              fmt::format("{}[flat_index] = {};", pad.second.name(), source);
          for_every_element(pad.first, pad.second, statement);
        }
      }
      emit("__syncthreads();");
    }

    if (leaf->type == SNodeType::dynamic) {
      emit("if ({} < {}_cache->get_n())", loopgen.loop_variable(leaf),
           leaf->node_type_name);
    }

    emit("{{");

    current_scratch_pads = scratch_pads.get();
    for_stmt->body->accept(this);
    current_scratch_pads = nullptr;

    emit("}}");

    bool needs_write_back = false;
    if (!for_stmt->scratch_opt.empty()) {
      for (auto &pad : scratch_pads->pads) {
        if (pad.second.total_flags == AccessFlag::write ||
            pad.second.total_flags == AccessFlag::accumulate) {
          needs_write_back = true;
        }
      }
    }

    if (needs_write_back) {
      emit("__syncthreads();");
      for (auto &pad : scratch_pads->pads) {
        if (pad.second.total_flags == AccessFlag::write ||
            pad.second.total_flags == AccessFlag::accumulate) {
          std::string source = access_global(pad.first);
          std::string statement;
          if (pad.second.total_flags == AccessFlag::accumulate)
            statement = fmt::format(
                "if ({}[flat_index] != 0) atomicAdd(&{}->val, "
                "{}[flat_index]);",
                pad.second.name(), source, pad.second.name());
          else
            statement =
                fmt::format("*{} = {}[flat_index];", source, pad.second.name());
          for_every_element(pad.first, pad.second, statement);
        }
      }
    }

    emit("}}");

    emit("extern \"C\" void {} (Context context) {{\n", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         current_program->snode_root->node_type_name);
    emit("{{");

    emit("auto t = get_time();");

    std::string vars;
    for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
      vars += for_stmt->loop_vars[i]->raw_name();
      if (i + 1 < for_stmt->loop_vars.size()) {
        vars += ",";
      }
    }
    emit("gpu_runtime_init();");
    emit("context.num_leaves = Managers::get_allocator<{}>()->resident_tail;",
         leaf->parent->node_type_name);

    emit("context.leaves = Managers::get_allocator<{}>()->meta_pool;",
         leaf->parent->node_type_name);

    emit(
        "int gridDim = context.num_leaves * {}, blockDim = ({}::get_max_n()"
        "+ {} - 1) / {};",
        block_division, leaf->node_type_name, block_division, block_division);
    emit(
        "printf(\"launching kernel {} <<<%d, %d>>> num_leaves = %d\\n\", "
        "gridDim, blockDim, context.num_leaves);",
        codegen->func_name);
    emit("cudaEvent_t start, stop;");

    if (current_program->get_current_kernel().benchmarking) {
      emit("while(1) {{");
    }

    emit("cudaEventCreate(&start);");
    emit("cudaEventCreate(&stop);");
    emit("cudaEventRecord(start);");
    emit("{}_kernel<<<gridDim, blockDim>>>(context);", codegen->func_name);
    emit("cudaEventRecord(stop);");

    // emit("t = get_time();");
    // emit("std::cout << (get_time() - t) * 1000 << std::endl;");
    emit("cudaEventSynchronize(stop);");

    emit("float milliseconds = 0;");
    emit("cudaEventElapsedTime(&milliseconds, start, stop);");
    emit("std::cout << \"device only \" << milliseconds << \" ms\" << std::endl;");

    if (current_program->current_kernel->benchmarking) {
      emit("cudaDeviceSynchronize();\n");
      emit("auto err = cudaGetLastError();");
      emit("if (err) {{");
      emit(
          "printf(\"CUDA Error (File %s Ln %d): %s\\n\", "
          "__FILE__, __LINE__, cudaGetErrorString(err));");
      emit("exit(-1);}}");
      emit("}}");
    }

    emit("context.leaves = nullptr;");
    emit("}}");
    current_struct_for = nullptr;
  }

  void generate_pure_loop(Block *stmt_list) {
    auto &for_stmt_ = stmt_list->statements.back();
    if (for_stmt_->is<RangeForStmt>()) {
      auto range_for = for_stmt_->as<RangeForStmt>();

      // GPU Kernel
      emit("__global__ void {}_kernel(Context context) {{", codegen->func_name);
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

      int block_size = range_for->block_size;
      if (block_size == 0) {
        TC_WARN("Using default block size = 256");
        block_size = 256;
      }
      emit("gpu_runtime_init();");
      int num_blocks = (end - begin + block_size - 1) / block_size;
      emit("{}_kernel<<<{}, {}>>>(context);", codegen->func_name, num_blocks,
           block_size);
    } else {
      auto struct_for = for_stmt_->as<StructForStmt>();
      bool use_activity_tracking = false;
      if (struct_for->snode->parent->type == SNodeType::dense &&
          struct_for->snode->parent->parent->type == SNodeType::pointer)
        use_activity_tracking = true;

      if (use_activity_tracking) {
        TC_WARN("Using activity tracking");
        struct_for_new(for_stmt_.get());
      } else {
        struct_for_old(for_stmt_.get());
      }
    }

    emit("cudaDeviceSynchronize();\n");
    emit("auto err = cudaGetLastError();");
    emit("if (err) {{");
    emit(
        "printf(\"CUDA Error (File %s Ln %d): %s\\n\", "
        "__FILE__, __LINE__, cudaGetErrorString(err));");
    emit("exit(-1);}}");
    emit("}}\n");
  }

  void visit(Block *stmt_list) {
    if (first_level) {
      first_level = false;
      // Check structure
      // Only the last statement can be a RangeFor/StructFor
      // The rest must be Alloca for loop variables and consts for bounds

      bool pure_loop = true;

      for (int i = 0; i + 1 < stmt_list->statements.size(); i++) {
        auto s = stmt_list->statements[i].get();
        if (!(s->is<AllocaStmt>() || s->is<ConstStmt>())) {
          pure_loop = false;
        }
      }

      if (pure_loop) {
        generate_pure_loop(stmt_list);
      } else {
        // GPU Kernel
        emit("__global__ void {}_kernel(Context context) {{",
             codegen->func_name);
        emit("auto root = ({} *)context.buffers[0];",
             codegen->prog->snode_root->node_type_name);

        stmt_list->accept(this);

        emit("}}\n\n");

        // CPU Kernel code
        emit("extern \"C\" void {} (Context context) {{\n", codegen->func_name);
        emit("gpu_runtime_init();");
        emit("{}_kernel<<<1, 1>>>(context);", codegen->func_name);
        emit("}}\n\n");
      }
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
    emit("const auto {} = randf();", stmt->raw_name(),
         stmt->ret_data_type_name());
  }

  void visit(UnaryOpStmt *stmt) {
    if (stmt->op_type != UnaryType::cast) {
      emit("const {} {}({}({}));", stmt->ret_data_type_name(), stmt->raw_name(),
           unary_type_name(stmt->op_type), stmt->rhs->raw_name());
    } else {
      if (stmt->cast_by_value)
        emit("const {} {}(static_cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->rhs->raw_name());
      else
        emit("const {} {}(union_cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->rhs->raw_name());
    }
  }

  void visit(BinaryOpStmt *bin) {
    std::string ns = "";
    if (bin->op_type == BinaryType::div) {
      ns = "taichi::Tlang::";
    }
    emit("const {} {}({}{}({}, {}));", bin->ret_data_type_name(),
         bin->raw_name(), ns, binary_type_name(bin->op_type),
         bin->lhs->raw_name(), bin->rhs->raw_name());
  }

  void visit(TrinaryOpStmt *tri) {
    TC_ASSERT(tri->op_type == TrinaryType::select);
    emit("const {} {} = {} ? {} : {};", tri->ret_data_type_name(),
         tri->raw_name(), tri->op1->raw_name(), tri->op2->raw_name(),
         tri->op3->raw_name());
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
    emit("if (!{}) break;", stmt->cond->raw_name());
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
    if (loop_var->ret_type.width == 1 &&
        loop_var->ret_type.data_type == DataType::i32) {
      emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
           loop_var->raw_name(), for_stmt->begin->raw_name(),
           loop_var->raw_name(), for_stmt->end->raw_name(),
           loop_var->raw_name(), loop_var->raw_name(), 1);
      emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
    } else {
      emit("for ({} {} = {}; {} < {}; {} = {} + {}({})) {{",
           loop_var->ret_data_type_name(), loop_var->raw_name(),
           for_stmt->begin->raw_name(), loop_var->raw_name(),
           for_stmt->end->raw_name(), loop_var->raw_name(),
           loop_var->raw_name(), loop_var->ret_data_type_name(), 1);
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
    emit("{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(GlobalPtrStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
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
              stmt->indices[i]->raw_name();
        }
      }
      std::string strong_access =
          fmt::format("{}[{}] = &access_{}{}->val;", stmt->raw_name(), l,
                      stmt->snodes[l]->node_type_name,
                      "(root, " + make_list(indices, "") + ")");

      emit("{}", strong_access);
    }
  }

  void visit(SNodeOpStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    auto snode = stmt->snodes[0];
    auto indices = indices_str(snode, -1, stmt->indices);

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
      emit("{}_tmp->append({}({}));", snode->node_type_name,
           snode->ch[0]->node_type_name, stmt->val->raw_name());
    } else if (stmt->op_type == SNodeOpType::clear) {
      emit("{}_tmp->clear();", snode->node_type_name);
    } else if (stmt->op_type == SNodeOpType::probe) {
      emit("{} = {}_tmp->get_n();", stmt->raw_name(), snode->node_type_name);
    } else if (stmt->op_type == SNodeOpType::activate) {
      emit("activate_{}(root, {});", snode->node_type_name,
           make_list(indices, ""));
    } else {
      TC_NOT_IMPLEMENTED;
    }
    emit("}}");
  }

  void visit(GlobalStoreStmt *stmt) {
    auto ptr = stmt->ptr->as<GlobalPtrStmt>();
    auto snode = ptr->snodes[0];
    if (current_scratch_pads && current_scratch_pads->has(snode)) {
      auto &pad = current_scratch_pads->get(snode);
      emit("{}[{}] = {};", pad.name(),
           pad.global_to_linearized_local(current_struct_for->loop_vars,
                                          ptr->indices),
           stmt->data->raw_name());
    } else {
      emit("*({} *){}[{}] = {};",
           data_type_name(stmt->data->ret_type.data_type),
           stmt->ptr->raw_name(), 0, stmt->data->raw_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    auto ptr = stmt->ptr->as<GlobalPtrStmt>();
    auto snode = ptr->snodes[0];
    if (current_scratch_pads && current_scratch_pads->has(snode)) {
      auto &pad = current_scratch_pads->get(snode);
      emit("const auto {} = {}[{}];", stmt->raw_name(), pad.name(),
           pad.global_to_linearized_local(current_struct_for->loop_vars,
                                          ptr->indices));
    } else {
      emit("const auto {} = *({}[0]);", stmt->raw_name(),
           stmt->ptr->raw_name());
    }
  }

  void visit(AtomicOpStmt *stmt) {
    TC_ASSERT(stmt->val->ret_type.data_type == DataType::f32 ||
              stmt->val->ret_type.data_type == DataType::i32);
    TC_ASSERT(stmt->op_type == AtomicType::add);
    auto ptr = stmt->dest->as<GlobalPtrStmt>();
    auto snode = ptr->snodes[0];
    if (current_scratch_pads && current_scratch_pads->has(snode)) {
      auto &pad = current_scratch_pads->get(snode);
      emit("atomicAdd(&{}[{}], {});", pad.name(),
           pad.global_to_linearized_local(current_struct_for->loop_vars,
                                          ptr->indices),
           stmt->val->raw_name());
    } else {
      emit("atomicAdd({}[0], {});", stmt->dest->raw_name(),
           stmt->val->raw_name());
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

  void visit(RangeAssumptionStmt *stmt) {
    emit("const auto {} = {};", stmt->raw_name(), stmt->input->raw_name());
  }
};

void GPUCodeGen::lower() {
  auto ir = kernel->ir;
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::lower(ir);
  irpass::re_id(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  irpass::re_id(ir);
  if (prog->config.print_ir) {
    irpass::print(ir);
  }
  irpass::eliminate_dup(ir);
  irpass::re_id(ir);
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
