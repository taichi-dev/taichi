// The CUDA backend

#include "codegen_cuda.h"
#include "loopgen.h"
#include "../scratch_pad.h"

TLANG_NAMESPACE_BEGIN

class GPUIRCodeGen : public IRVisitor {
 public:
  StructForStmt *current_struct_for;
  ScratchPads *current_scratch_pads;
  GPUCodeGen *codegen;
  LoopGenerator loopgen;
  bool first_level = false;
  bool debug;
  int grid_dim;
  int for_stmt_counter;
  CompileConfig cfg;
  std::set<SNode *> ldg;

  GPUIRCodeGen(GPUCodeGen *codegen) : codegen(codegen), loopgen(codegen) {
    current_struct_for = nullptr;
    current_scratch_pads = nullptr;
    cfg = codegen->prog->config;
    debug = cfg.debug;
    grid_dim = loopgen.grid_dim;
    for_stmt_counter = 0;
  }

  template <typename... Args>
  void emit(std::string f, Args &&... args) {
    codegen->emit(f, std::forward<Args>(args)...);
  }

  static void run(GPUCodeGen *codegen, IRNode *node) {
    auto p = GPUIRCodeGen(codegen);
    p.first_level = true;
    CODE_REGION(body);
    node->accept(&p);
  }

  void struct_for(Stmt *for_stmt_) {
    TI_ASSERT_INFO(current_struct_for == nullptr,
                   "Struct for cannot be nested.");
    auto for_stmt = for_stmt_->as<StructForStmt>();
    current_struct_for = for_stmt;
    auto leaf = for_stmt->snode->parent;

    if (for_stmt->block_dim == 0) {
      for_stmt->block_dim =
          std::min(1 << leaf->total_num_bits, max_gpu_block_dim);
    }

    TI_ASSERT((1 << leaf->total_num_bits) % for_stmt->block_dim == 0);
    int block_division = (1 << leaf->total_num_bits) / for_stmt->block_dim;

    std::vector<SNode *> path;
    for (auto p = leaf; !p->has_allocator(); p = p->parent) {
      path.push_back(p);
    }

    auto listgen_suffix = fmt::format("{}", for_stmt_counter);
    {
      CODE_REGION(gpu_kernels);

      loopgen.emit_listgen_func(leaf, block_division, listgen_suffix);

      for (int i = 1; i < (int)path.size(); i++) {
        loopgen.emit_listgen_func(path[i], 0, listgen_suffix);
      }

      emit("__global__ void {}_kernel(Context context) {{",
           current_func_name());
      emit("auto root = ({} *)context.buffers[0];",
           codegen->prog->snode_root->node_type_name);

      emit(
          "auto leaves = (SNodeMeta "
          "*)(Managers::get_allocator<{}>()->resident_pool);",
          leaf->node_type_name);
      // TODO: use const tail
      emit("auto num_leaves = Managers::get_allocator<{}>()->resident_tail;",
           leaf->node_type_name);
      emit("int bid = 0;");
      emit("while (1) {{");
      emit("__syncthreads();");  // This syncthread is necessary otherwise
                                 // bid_shared may be overwritten..
      emit("__shared__ int bid_shared;");
      emit("if (threadIdx.x == 0) {{ ");
      emit(
          "bid_shared = atomicAdd((unsigned long long "
          "*)(&Managers::get_allocator<{}>()->execution_tail), 1ULL);",
          leaf->node_type_name);
      emit("}}");

      emit("__syncthreads();");
      emit("bid = bid_shared;");
      emit("if (bid >= num_leaves) break;");
      emit("auto leaf_loop = bid;");

      emit("auto list_element = ({} *)leaves[leaf_loop].ptr;",
           leaf->node_type_name);
      auto chid = leaf->parent->child_id(leaf);

      emit("auto {}_cache = list_element;", leaf->node_type_name,
           leaf->node_type_name);
      for (int i = 0; i < max_num_indices; i++) {
        emit("auto {} = leaves[leaf_loop].indices[{}];",
             loopgen.index_name_global(leaf->parent, i), i);
      }
      emit("auto {} = threadIdx.x + leaves[leaf_loop].start_loop;",
           loopgen.loop_variable(leaf));

      loopgen.update_indices(leaf);
      loopgen.emit_setup_loop_variables(for_stmt, leaf);

      std::unique_ptr<ScratchPads> scratch_pads;

      auto access_global = [&](SNode *snode) -> std::string {
        std::vector<std::string> indices(max_num_indices, "0");
        for (int i = 0; i < (int)for_stmt->loop_vars.size(); i++) {
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
        emit("__syncthreads();");
        scratch_pads = irpass::initialize_scratch_pad(for_stmt);
        for (auto &pad : scratch_pads->pads) {
          emit("__shared__ {}::val_type {}[{}];", pad.first->node_type_name,
               pad.second.name(), pad.second.linear_size());
          TI_ASSERT(pad.second.is_pure());
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
        emit("if ({} < leaves[leaf_loop].end_loop)",
             loopgen.loop_variable(leaf), leaf->node_type_name);
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
              statement = fmt::format("*{} = {}[flat_index];", source,
                                      pad.second.name());
            for_every_element(pad.first, pad.second, statement);
          }
        }
      }

      emit("}}");  // end for
      emit("}}");  // end kernel
    }

    {
      CODE_REGION(body);

      emit("{{");
      if (debug)
        emit("auto t = get_time();");

      std::string vars;
      for (int i = 0; i < (int)for_stmt->loop_vars.size(); i++) {
        vars += for_stmt->loop_vars[i]->raw_name();
        if (i + 1 < (int)for_stmt->loop_vars.size()) {
          vars += ",";
        }
      }
      emit("gpu_runtime_init();");
      emit("int blockDim = ({}::get_max_n()+ {} - 1) / {};",
           leaf->node_type_name, block_division, block_division);
      emit("");

      // generate the list
      if (cfg.enable_profiler)
        emit(R"(CUDAProfiler::get_instance().start("{}_list_gen");)",
             codegen->func_name);

      std::reverse(path.begin(), path.end());
      for (auto &s : path) {
        emit("{}(context);", loopgen.listgen_func_name(s, listgen_suffix));
      }
      if (cfg.enable_profiler)
        emit(R"(CUDAProfiler::get_instance().stop();)");

      emit("");

      if (debug) {
        emit("cudaDeviceSynchronize();");
        emit(
            R"(printf("task list %d\n", Managers::get_allocator<{}>()->resident_tail);)",
            leaf->node_type_name);
        emit(R"(printf("kernel {} <<<%d, %d>>> \n", {}, blockDim);)",
             codegen->func_name, grid_dim);

        emit("cudaEvent_t start, stop;");

        if (current_program->get_current_kernel().benchmarking) {
          emit("while(1) {{");
        }

        emit("cudaEventCreate(&start);");
        emit("cudaEventCreate(&stop);");
        emit("cudaEventRecord(start);");
      }
      emit("");
      emit("reset_execution_tail<{}><<<1, 1>>>();", leaf->node_type_name);
      if (cfg.enable_profiler)
        emit(R"(CUDAProfiler::get_instance().start("{}");)",
             current_func_name());
      emit("{}_kernel<<<{}, blockDim>>>(context);", current_func_name(),
           grid_dim);
      if (cfg.enable_profiler)
        emit(R"(CUDAProfiler::get_instance().stop();)");
      emit("");
      if (debug) {
        emit("cudaEventRecord(stop);");
        emit("cudaEventSynchronize(stop);");

        emit("float milliseconds = 0;");
        emit("cudaEventElapsedTime(&milliseconds, start, stop);");
        emit(
            R"(std::cout << "     device only : " << milliseconds << " ms\n";)");

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
      }
      emit("}}");
      current_struct_for = nullptr;
    }
  }

  void extract_ldg(ScratchPadOptions &opt) {
    ScratchPadOptions new_opt;
    for (auto &o : opt) {
      if (o.first == 1) {
        ldg.insert(o.second);
        TI_INFO("Caching to L1: {}", o.second->node_type_name);
      } else {
        new_opt.push_back(o);
      }
    }
    opt = new_opt;
  }

  std::string current_func_name() {
    return fmt::format(codegen->func_name +
                       fmt::format("_{}", for_stmt_counter));
  }

  // For cases where the kernel body has only for loops
  // TODO: rename
  void generate_pure_loop(Stmt *for_stmt_) {
    for_stmt_counter += 1;
    if (for_stmt_->is<RangeForStmt>()) {
      auto range_for = for_stmt_->as<RangeForStmt>();
      TI_ASSERT(range_for->vectorize == 1);

      int begin = range_for->begin->as<ConstStmt>()->val[0].val_int32();
      int end = range_for->end->as<ConstStmt>()->val[0].val_int32();
      {
        // GPU Kernel
        CODE_REGION(gpu_kernels);

        emit("__global__ void {}_kernel(Context context) {{",
             current_func_name());
        emit("auto root = ({} *)context.buffers[0];",
             codegen->prog->snode_root->node_type_name);

        emit("auto {} = blockIdx.x * blockDim.x + threadIdx.x + {};\n",
             range_for->loop_var->raw_name(), begin);
        emit("if ({} >= {}) return;", range_for->loop_var->raw_name(), end);

        range_for->body->accept(this);

        emit("}}\n\n");
      }

      // host code
      emit("{{");
      int block_dim = range_for->block_dim;
      if (block_dim == 0) {
        TI_WARN("Using default block size = 256");
        block_dim = 256;
      }
      emit("gpu_runtime_init();");
      int num_blocks = (end - begin + block_dim - 1) / block_dim;
      if (cfg.enable_profiler)
        emit(R"(CUDAProfiler::get_instance().start("{}");)", codegen->func_name);
      emit("{}_kernel<<<{}, {}>>>(context);", current_func_name(), num_blocks,
           block_dim);
      if (cfg.enable_profiler)
        emit(R"(CUDAProfiler::get_instance().stop();)");
      emit("}}");
    } else {
      auto for_stmt = for_stmt_->as<StructForStmt>();
      TI_ASSERT(for_stmt->vectorize == 1);
      extract_ldg(for_stmt->scratch_opt);
      struct_for(for_stmt_);
    }

    if (debug) {
      emit("cudaDeviceSynchronize();\n");
      emit(
          R"(if (allocator()->gpu_error_code) {{printf("GPU Assertion Error\n"); exit(-1);}})");
      emit("auto err = cudaGetLastError();");
      emit("if (err) {{");
      emit(
          "printf(\"CUDA Error (File %s Ln %d): %s\\n\", "
          "__FILE__, __LINE__, cudaGetErrorString(err));");
      emit("exit(-1);}}");
    }
  }

  void visit(Block *stmt_list) override {
    if (first_level) {
      first_level = false;
      // Check structure
      // Only the last statement can be a RangeFor/StructFor
      // The rest must be Alloca for loop variables and consts for bounds

      bool pure_loop = true;

      for (int i = 0; i < (int)stmt_list->statements.size(); i++) {
        auto s = stmt_list->statements[i].get();
        if (!(s->is<AllocaStmt>() || s->is<ConstStmt>() ||
              s->is<StructForStmt>() || s->is<RangeForStmt>())) {
          pure_loop = false;
        }
      }
      bool is_clearer = false;
      if (stmt_list->statements.back()->is<ClearAllStmt>())
        is_clearer = true;

      if (is_clearer) {
        for (auto &stmt : stmt_list->statements) {
          stmt->accept(this);
        }
      } else if (pure_loop) {
        emit("extern \"C\" void {} (Context &context) {{\n",
             codegen->func_name);
        emit("auto root = ({} *)context.buffers[0];",
             current_program->snode_root->node_type_name);
        for (int i = 0; i < (int)stmt_list->statements.size(); i++) {
          auto s = stmt_list->statements[i].get();
          if (s->is<StructForStmt>() || s->is<RangeForStmt>()) {
            generate_pure_loop(s);
          }
        }
        emit("}}");
      } else {
        // Single threaded GPU Kernel
        emit("__global__ void {}_kernel(Context context) {{",
             codegen->func_name);
        emit("auto root = ({} *)context.buffers[0];",
             codegen->prog->snode_root->node_type_name);

        stmt_list->accept(this);

        emit("}}\n\n");

        // CPU Kernel code
        emit("extern \"C\" void {} (Context &context) {{\n",
             codegen->func_name);
        emit("gpu_runtime_init();");

        if (cfg.enable_profiler)
          emit(R"(CUDAProfiler::get_instance().start("{}");)",
               current_func_name());
        emit("{}_kernel<<<1, 1>>>(context);", codegen->func_name);
        if (cfg.enable_profiler)
          emit(R"(CUDAProfiler::get_instance().stop();)");
        emit("}}\n\n");
      }
    } else {
      for (auto &stmt : stmt_list->statements) {
        stmt->accept(this);
      }
    }
  }

  void visit(AllocaStmt *alloca) override {
    emit("{} {}(0);", alloca->ret_data_type_name(), alloca->raw_name());
  }

  void visit(RandStmt *stmt) override {
    TI_ASSERT(stmt->ret_type.data_type == DataType::f32);
    emit("const auto {} = randf();", stmt->raw_name(),
         stmt->ret_data_type_name());
  }

  void visit(UnaryOpStmt *stmt) override {
    if (stmt->op_type != UnaryOpType::cast) {
      emit("const {} {}({}({}));", stmt->ret_data_type_name(), stmt->raw_name(),
           unary_op_type_name(stmt->op_type), stmt->operand->raw_name());
    } else {
      if (stmt->cast_by_value)
        emit("const {} {}(static_cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->operand->raw_name());
      else
        emit("const {} {}(union_cast<{}>({}));", stmt->ret_data_type_name(),
             stmt->raw_name(), data_type_name(stmt->cast_type),
             stmt->operand->raw_name());
    }
  }

  void visit(BinaryOpStmt *bin) override {
    std::string ns;
    if (bin->op_type == BinaryOpType::div) {
      ns = "taichi::Tlang::";
    }
    emit("const {} {}({}{}({}, {}));", bin->ret_data_type_name(),
         bin->raw_name(), ns, binary_op_type_name(bin->op_type),
         bin->lhs->raw_name(), bin->rhs->raw_name());
  }

  void visit(TernaryOpStmt *tri) override {
    TI_ASSERT(tri->op_type == TernaryOpType::select);
    emit("const {} {} = {} ? {} : {};", tri->ret_data_type_name(),
         tri->raw_name(), tri->op1->raw_name(), tri->op2->raw_name(),
         tri->op3->raw_name());
  }

  void visit(IfStmt *if_stmt) override {
    emit("if ({}) {{", if_stmt->cond->raw_name());
    if (if_stmt->true_statements)
      if_stmt->true_statements->accept(this);
    if (if_stmt->false_statements) {
      emit("}} else {{");
      if_stmt->false_statements->accept(this);
    }
    emit("}}");
  }

  void visit(PrintStmt *print_stmt) override {
    if (print_stmt->stmt->ret_type.data_type == DataType::i32) {
      emit("printf(\"[debug] {}\" \" = %d\\n\", {});", print_stmt->str,
           print_stmt->stmt->raw_name());
    } else if (print_stmt->stmt->ret_type.data_type == DataType::f32) {
      emit("printf(\"[debug] {}\" \" = %f\\n\", {});", print_stmt->str,
           print_stmt->stmt->raw_name());
    } else if (print_stmt->stmt->ret_type.data_type == DataType::f64) {
      emit("printf(\"[debug] {}\" \" = %f\\n\", {});", print_stmt->str,
           print_stmt->stmt->raw_name());
    } else {
      TI_NOT_IMPLEMENTED
    }
  }

  void visit(ConstStmt *const_stmt) override {
    emit("const {} {}({});", const_stmt->ret_type.str(), const_stmt->raw_name(),
         const_stmt->val.serialize(
             [&](const TypedConstant &t) { return t.stringify(); }, "{"));
  }

  void visit(WhileControlStmt *stmt) override {
    emit("if (!{}) break;", stmt->cond->raw_name());
  }

  void visit(WhileStmt *stmt) override {
    emit("while (1) {{");
    stmt->body->accept(this);
    emit("}}");
  }

  void visit(StructForStmt *for_stmt) override {
    // generate_loop_header(for_stmt->snode, for_stmt, true);
    TI_ASSERT_INFO(current_struct_for == nullptr,
                   "StructFor cannot be nested.");
    current_struct_for = for_stmt;
    for_stmt->body->accept(this);
    current_struct_for = nullptr;
    // generate_loop_tail(for_stmt->snode, for_stmt, true);
  }

  void visit(RangeForStmt *for_stmt) override {
    auto loop_var = for_stmt->loop_var;
    if (loop_var->ret_type.width == 1 &&
        loop_var->ret_type.data_type == DataType::i32) {
      if (!for_stmt->reversed) {
        emit("for (int {}_ = {}; {}_ < {}; {}_ = {}_ + {}) {{",
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), 1);
        emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      } else {
        // reversed for loop
        emit("for (int {}_ = {} - 1; {}_ >= {}; {}_ = {}_ - {}) {{",
             loop_var->raw_name(), for_stmt->end->raw_name(),
             loop_var->raw_name(), for_stmt->begin->raw_name(),
             loop_var->raw_name(), loop_var->raw_name(), 1);
        emit("{} = {}_;", loop_var->raw_name(), loop_var->raw_name());
      }
    } else {
      TI_ASSERT(!for_stmt->reversed);
      emit("for ({} {} = {}; {} < {}; {} = {} + {}({})) {{",
           loop_var->ret_data_type_name(), loop_var->raw_name(),
           for_stmt->begin->raw_name(), loop_var->raw_name(),
           for_stmt->end->raw_name(), loop_var->raw_name(),
           loop_var->raw_name(), loop_var->ret_data_type_name(), 1);
    }
    for_stmt->body->accept(this);
    emit("}}");
  }

  void visit(ArgLoadStmt *stmt) override {
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

  void visit(LocalLoadStmt *stmt) override {
    auto ptr = stmt->ptr[0].var;
    emit("const {} {}({});", stmt->ret_data_type_name(), stmt->raw_name(),
         ptr->raw_name());
  }

  void visit(LocalStoreStmt *stmt) override {
    emit("{} = {};", stmt->ptr->raw_name(), stmt->data->raw_name());
  }

  void visit(ExternalPtrStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    TI_ASSERT(stmt->indices.size() == 1);
    auto dt = stmt->ret_type.data_type;
    emit("const {} *{}[1] = {{&{}[{}]}};", data_type_name(dt), stmt->raw_name(),
         stmt->base_ptrs[0]->raw_name(), stmt->indices[0]->raw_name());
  }

  void visit(GlobalPtrStmt *stmt) override {
    auto snode = stmt->snodes[0];
    if (current_scratch_pads && current_scratch_pads->has(snode)) {
      return;
    }

    TI_ASSERT(stmt->width() == 1);
    emit("{} *{}[{}];", data_type_name(stmt->ret_type.data_type),
         stmt->raw_name(), stmt->ret_type.width);
    for (int l = 0; l < stmt->ret_type.width; l++) {
      // Try to weaken here...
      std::vector<int> offsets(stmt->indices.size());

      auto snode = stmt->snodes[l];
      std::vector<std::string> indices(max_num_indices, "0");  // = "(root, ";
      for (int i = 0; i < (int)stmt->indices.size(); i++) {
        if (snode->physical_index_position[i] != -1) {
          // TI_ASSERT(snode->physical_index_position[i] != -1);
          indices[snode->physical_index_position[i]] =
              stmt->indices[i]->raw_name();
        }
      }
      std::string strong_access = fmt::format(
          "{}[{}] = &{}_{}{}->val;", stmt->raw_name(), l,
          stmt->accessor_func_name(), stmt->snodes[l]->node_type_name,
          "(root, " + make_list(indices, "") + ")");

      emit("{}", strong_access);
    }
  }

  void visit(SNodeOpStmt *stmt) override {
    /*
    TI_ASSERT(stmt->width() == 1);
    auto snode = stmt->snodes[0];
    auto indices = indices_str(snode, -1, stmt->indices);

    if (stmt->op_type == SNodeOpType::length) {
      emit("int32x1 {};", stmt->raw_name());
    }

    emit("{{");
    if (stmt->op_type != SNodeOpType::activate &&
        stmt->op_type != SNodeOpType::length)
      emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
           snode->node_type_name, snode->node_type_name,
           make_list(indices, ""));

    if (stmt->op_type == SNodeOpType::append) {
      TI_ASSERT(stmt->val->width() == 1);
      emit("{}_tmp->append({}({}));", snode->node_type_name,
           snode->ch[0]->node_type_name, stmt->val->raw_name());
    } else if (stmt->op_type == SNodeOpType::clear) {
      emit("{}_tmp->clear();", snode->node_type_name);
    } else if (stmt->op_type == SNodeOpType::length) {
      emit("{} = query_{}(root, {});", stmt->raw_name(), snode->node_type_name,
           make_list(indices, ""));
      if (snode->type == SNodeType::dynamic) {
        emit("if ({}) {{", stmt->raw_name());
        emit("{} *{}_tmp = access_{}(root, {});", snode->node_type_name,
             snode->node_type_name, snode->node_type_name,
             make_list(indices, ""));
        emit("{} = {}_tmp->get_n();", stmt->raw_name(), snode->node_type_name);
        emit("}}");
      }
    } else if (stmt->op_type == SNodeOpType::activate) {
      emit("activate_{}(root, {});", snode->node_type_name,
           make_list(indices, ""));
    } else {
      TI_NOT_IMPLEMENTED;
    }
    emit("}}");
    */
  }

  void visit(GlobalStoreStmt *stmt) override {
    if (stmt->ptr->is<GlobalPtrStmt>()) {
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
    } else {
      emit("*({} *){}[{}] = {};",
           data_type_name(stmt->data->ret_type.data_type),
           stmt->ptr->raw_name(), 0, stmt->data->raw_name());
    }
  }

  void visit(GlobalLoadStmt *stmt) override {
    TI_ASSERT(stmt->width() == 1);
    if (stmt->ptr->is<GlobalPtrStmt>()) {
      auto ptr = stmt->ptr->as<GlobalPtrStmt>();
      auto snode = ptr->snodes[0];
      if (current_scratch_pads && current_scratch_pads->has(snode)) {
        auto &pad = current_scratch_pads->get(snode);
        emit("const auto {} = {}[{}];", stmt->raw_name(), pad.name(),
             pad.global_to_linearized_local(current_struct_for->loop_vars,
                                            ptr->indices));
      } else {
        if (ldg.find(snode) != ldg.end()) {
          emit("const auto {} = __ldg({}[0]);", stmt->raw_name(),
               stmt->ptr->raw_name());
        } else {
          emit("const auto {} = *({}[0]);", stmt->raw_name(),
               stmt->ptr->raw_name());
        }
      }
    } else {
      SNode *snode = nullptr;
      if (stmt->ptr->is<GetChStmt>()) {
        auto ptr = stmt->ptr->as<GetChStmt>();
        snode = ptr->output_snode;
      } else if (stmt->ptr->is<IntegerOffsetStmt>()) {
        auto ptr = stmt->ptr->as<IntegerOffsetStmt>();
        snode = ptr->input->as<GetChStmt>()->output_snode;
      }
      if (ldg.find(snode) != ldg.end()) {
        emit("const auto {} = __ldg({}[0]);", stmt->raw_name(),
             stmt->ptr->raw_name());
      } else {
        emit("const auto {} = *({}[0]);", stmt->raw_name(),
             stmt->ptr->raw_name());
      }
    }
  }

  void visit(AtomicOpStmt *stmt) override {
    TI_ASSERT(stmt->val->ret_type.data_type == DataType::f32 ||
              stmt->val->ret_type.data_type == DataType::i32 ||
              stmt->val->ret_type.data_type == DataType::f64 ||
              stmt->val->ret_type.data_type == DataType::i64);
    TI_ASSERT(stmt->op_type == AtomicOpType::add);
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

  void visit(ElementShuffleStmt *stmt) override {
    auto init = stmt->elements.serialize(
        [&](const VectorElement &elem) {
          TI_ASSERT(elem.index == 0);
          if (stmt->pointer) {
            return fmt::format("{}[0]", elem.stmt->raw_name(), elem.index);
          } else {
            return fmt::format("{}", elem.stmt->raw_name(), elem.index);
          }
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

  void visit(RangeAssumptionStmt *stmt) override {
    // this does not necessarily hold since any index within the leaf block can
    // be the base
    /*
    emit("TI_ASSERT({} + {} <= {});", stmt->base->raw_name(), stmt->low,
         stmt->input->raw_name());
    emit("TI_ASSERT({} < {} + {});", stmt->input->raw_name(),
         stmt->base->raw_name(), stmt->high);
         */
    emit("const auto {} = {};", stmt->raw_name(), stmt->input->raw_name());
  }

  void visit(AssertStmt *stmt) override {
    emit("#if defined(TL_DEBUG)");
    emit(R"(TI_ASSERT_INFO({}, "{}");)", stmt->val->raw_name(), stmt->text);
    emit("#endif");
  }

  void visit(OffsetAndExtractBitsStmt *stmt) override {
    emit(R"(auto {} = ((({} + {}) >> {}) & ((1 << {}) - 1));)",
         stmt->raw_name(), stmt->offset, stmt->input->raw_name(),
         stmt->bit_begin, stmt->bit_end - stmt->bit_begin);
  }

  void visit(LinearizeStmt *stmt) override {
    std::string val = "0";
    for (int i = 0; i < (int)stmt->inputs.size(); i++) {
      val = fmt::format("({}) * {} + {}", val, stmt->strides[i],
                        stmt->inputs[i]->raw_name());
    }
    emit(R"(auto {} = {};)", stmt->raw_name(), val);
  }

  void visit(IntegerOffsetStmt *stmt) override {
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

  void visit(SNodeLookupStmt *stmt) override {
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
            stmt->global_indices[i]->raw_name();
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
      emit(
          "if({}_guarded == nullptr) {}_guarded = "
          "Managers::get_allocator<{}>()->ambient;",
          stmt->raw_name(), stmt->raw_name(), snode->node_type_name);
    }
    emit(R"(auto {} = {}_guarded;)", stmt->raw_name(), stmt->raw_name());
  }

  void visit(GetChStmt *stmt) override {
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

  void visit(ClearAllStmt *stmt) override {
    auto snode = stmt->snode;
    auto leaf = snode;

    std::vector<SNode *> path;
    for (auto p = leaf; !p->has_allocator(); p = p->parent) {
      path.push_back(p);
    }

    loopgen.emit_listgen_func(snode, 0, "", stmt->deactivate);
    for (int i = 1; i < (int)path.size(); i++) {
      loopgen.emit_listgen_func(path[i]);
    }

    emit("__global__ void {}_kernel(Context context) {{", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         codegen->prog->snode_root->node_type_name);

    emit(
        "auto leaves = (SNodeMeta "
        "*)(Managers::get_allocator<{}>()->resident_pool);",
        leaf->node_type_name);
    // TODO: use const tail
    emit("auto num_leaves = Managers::get_allocator<{}>()->resident_tail;",
         leaf->node_type_name);
    emit("int bid = 0;");
    emit("while (1) {{");
    emit("__shared__ int bid_shared;");
    emit("if (threadIdx.x == 0) {{ ");
    emit(
        "bid_shared = atomicAdd((unsigned long long "
        "*)(&Managers::get_allocator<{}>()->execution_tail), 1ULL);",
        leaf->node_type_name);
    emit("}}");

    emit("__syncthreads();");
    emit("bid = bid_shared;");
    emit("if (bid >= num_leaves) break;");
    emit("auto leaf_loop = bid;");

    emit("auto list_element = ({} *)leaves[leaf_loop].ptr;",
         leaf->node_type_name);
    emit("auto input_meta = leaves[leaf_loop];", leaf->node_type_name);
    auto chid = leaf->parent->child_id(leaf);

    emit("auto {}_cache = list_element;", leaf->node_type_name,
         leaf->node_type_name);
    for (int i = 0; i < max_num_indices; i++) {
      emit("auto {} = leaves[leaf_loop].indices[{}];",
           loopgen.index_name_global(leaf->parent, i), i);
    }
    emit(
        "for (int cid = input_meta.start_loop; cid < input_meta.end_loop; "
        "cid++) {{");
    emit("if (!{}_cache->is_active(cid)) continue;", snode->node_type_name);
    emit("auto {}_child = {}_cache->look_up(cid);", snode->node_type_name,
         snode->node_type_name);
    emit(
        "for(int i = threadIdx.x; i * sizeof(int32) < sizeof({}::child_type); "
        "i += "
        "blockDim.x) ",
        snode->node_type_name);

    emit("{{");

    emit("((int32 *){}_child)[i] = 0;", snode->node_type_name);

    emit("}}");

    emit("}}");

    emit("}}");  // end for
    emit("}}");  // end kernel

    emit("extern \"C\" void {} (Context &context) {{\n", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         current_program->snode_root->node_type_name);
    emit("{{");

    emit("gpu_runtime_init();");
    emit("int blockDim = {};", max_gpu_block_dim);
    emit("");

    // generate the list
    if (cfg.enable_profiler)
      emit(R"(CUDAProfiler::get_instance().start("{}_list_gen");)",
           codegen->func_name);

    std::reverse(path.begin(), path.end());
    for (auto &s : path) {
      emit("{}(context);", loopgen.listgen_func_name(s));
    }
    if (cfg.enable_profiler)
      emit(R"(CUDAProfiler::get_instance().stop();)");

    emit("");

    emit("reset_execution_tail<{}><<<1, 1>>>();", leaf->node_type_name);
    if (cfg.enable_profiler)
      emit(R"(CUDAProfiler::get_instance().start("clear_{}");)",
           snode->node_type_name);
    emit("{}_kernel<<<{}, blockDim>>>(context);", codegen->func_name, grid_dim);
    if (cfg.enable_profiler)
      emit(R"(CUDAProfiler::get_instance().stop();)");

    emit("");
    emit("}}");
    emit("}}");
  }
};

void GPUCodeGen::lower_cuda() {
  TI_NOT_IMPLEMENTED
}

void GPUCodeGen::lower_llvm() {
  auto ir = kernel->ir;
  bool print_ir = false;
  if (kernel->is_accessor) {
    print_ir = prog->config.print_accessor_ir;
  } else {
    print_ir = prog->config.print_ir;
  }
  if (print_ir) {
    irpass::print(ir);
  }
  if (kernel->grad) {
    irpass::reverse_segments(ir);
    if (print_ir) {
      TI_TRACE("Segment reversed (for autodiff):");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::lower(ir);
  irpass::re_id(ir);
  if (print_ir) {
    irpass::print(ir);
  }
  irpass::typecheck(ir);
  irpass::re_id(ir);
  if (print_ir) {
    irpass::print(ir);
  }
  if (prog->config.demote_dense_struct_fors) {
    irpass::demote_dense_struct_fors(ir);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_TRACE("Dense Struct-for demoted:");
      irpass::print(ir);
    }
  }
  irpass::constant_fold(ir);
  if (prog->config.simplify_before_lower_access) {
    irpass::simplify(ir);
    irpass::re_id(ir);
    if (print_ir) {
      TI_TRACE("Simplified I:");
      irpass::print(ir);
    }
  }
  if (kernel->grad) {
    // irpass::re_id(ir);
    // TI_TRACE("Primal:");
    // irpass::print(ir);
    irpass::demote_atomics(ir);
    irpass::full_simplify(ir, prog->config);
    irpass::typecheck(ir);
    if (print_ir) {
      TI_TRACE("Before make_adjoint:");
      irpass::print(ir);
    }
    irpass::make_adjoint(ir);
    if (print_ir) {
      TI_TRACE("After make_adjoint:");
      irpass::print(ir);
    }
    irpass::typecheck(ir);
    // irpass::re_id(ir);
    // TI_TRACE("Adjoint:");
    // irpass::print(ir);
  }
  irpass::lower_access(ir, prog->config.use_llvm);
  if (print_ir) {
    TI_TRACE("Access Lowered:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  if (prog->config.simplify_after_lower_access) {
    irpass::die(ir);
    if (print_ir) {
      TI_TRACE("DIEd:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    irpass::simplify(ir);
    if (print_ir) {
      TI_TRACE("Simplified II:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
  }
  irpass::die(ir);
  if (print_ir) {
    TI_TRACE("DIEd:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::flag_access(ir);
  if (print_ir) {
    TI_TRACE("Access Flagged:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::offload(ir);
  if (print_ir) {
    TI_TRACE("Offloaded:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::demote_atomics(ir);
  if (print_ir) {
    TI_TRACE("Atomics Demoted:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
  irpass::full_simplify(ir, prog->config);
  if (print_ir) {
    TI_TRACE("Simplified III:");
    irpass::re_id(ir);
    irpass::print(ir);
  }
}

void GPUCodeGen::lower() {
  TI_PROFILER(__FUNCTION__)
  if (prog->config.use_llvm) {
    lower_llvm();
  } else {
    lower_cuda();
  }
}

void GPUCodeGen::codegen() {
  emit("#define TI_GPU");
  generate_header();

  // Body
  GPUIRCodeGen::run(this, kernel->ir);

  line_suffix = "";
  generate_tail();
}

TLANG_NAMESPACE_END
