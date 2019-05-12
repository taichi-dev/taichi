#include <cuda_runtime.h>
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
  bool debug;
  int grid_dim;

  GPUIRCodeGen(GPUCodeGen *codegen) : codegen(codegen), loopgen(codegen) {
    current_struct_for = nullptr;
    current_scratch_pads = nullptr;
    debug = codegen->prog->config.debug;
    int num_SMs;
    cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
    grid_dim = num_SMs * 32;  // each SM can have 16-32 resident blocks
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

  void struct_for_old(Stmt *for_stmt_);

  void emit_listgen_func(SNode *snode, int child_block_division = 0) {
    auto child_block_size = 1 << snode->total_num_bits;
    if (child_block_division == 0) {
      // how many divisions should the CHILD node have?
      if (child_block_size > max_gpu_block_size) {
        child_block_division = child_block_size / max_gpu_block_size;
        child_block_size = max_gpu_block_size;
      } else {
        child_block_division = 1;
      }
    } else {
      child_block_size /= child_block_division;
    }
    // TC_P(child_block_size);
    // TC_P(child_block_division);

    auto parent = snode->parent;

    int parent_branching = 1;
    if (parent->type == SNodeType::dense) {  // TODO: dynamic
      parent_branching = 1 << parent->total_num_bits;
    }

    int listgen_block_dim =
        std::min(max_gpu_block_size, parent_branching * child_block_division);

    auto leaf_allocator =
        fmt::format("Managers::get_allocator<{}>()", snode->node_type_name);

    // kernel body starts
    emit("__global__ void {}_listgen_device(Context context) {{",
         snode->node_type_name);

    emit("int num_leaves = Managers::get_allocator<{}>()->resident_tail;",
         parent->node_type_name);
    emit("constexpr int parent_branching = {};", parent_branching);
    emit("while (1) {{");

    // one block takes one ancestor meta
    emit("__shared__ int bid_shared;");
    emit("if (threadIdx.x == 0) {{ ");
    emit(
        "bid_shared = atomicAdd((unsigned long long "
        "*)(&Managers::get_allocator<{}>()->execution_tail), 1ULL);",
        parent->node_type_name);
    emit("}}");

    emit("__syncthreads();");
    emit("int leaf_loop = bid_shared;");
    emit("if (leaf_loop >= num_leaves) break;");

    /*
    NOTE: when parent has an allocator, this kernel loops over the pointers
    to its child_type; otherwise pointers to the nodes themselves.

    */
    emit(
        "auto leaves = (SNodeMeta "
        "*)(Managers::get_allocator<{}>()->resident_pool);",
        parent->node_type_name);
    emit(
        "auto num_leaves = Managers::get_allocator<{}>()->resident_tail_const;",
        parent->node_type_name);

    emit("constexpr int child_block_division = {};", child_block_division);
    emit(
        "constexpr int num_divided_child_blocks = child_block_division * "
        "parent_branching;");

    if (parent->type == SNodeType::dense) {
      emit("auto &input_meta = leaves[leaf_loop];");
      emit("auto {}_cache = ({} *)input_meta.ptr;", parent->node_type_name,
           parent->node_type_name);
      for (int i = 0; i < max_num_indices; i++) {
        emit("auto {} = leaves[leaf_loop].indices[{}];",
             loopgen.index_name_global(parent->parent, i), i);
      }
    } else {
      emit("auto list_element = ({}::child_type *)leaves[leaf_loop].ptr;",
           parent->node_type_name);
      auto chid = parent->child_id(snode);
      TC_ASSERT(chid != -1);
      emit("auto {}_cache = list_element->get{}();", snode->node_type_name,
           chid);
      for (int i = 0; i < max_num_indices; i++) {
        emit("auto {} = leaves[leaf_loop].indices[{}];",
             loopgen.index_name_global(parent, i), i);
      }
    }
    emit(
        "for (int div = threadIdx.x; div < num_divided_child_blocks; div += "
        "{}) {{",
        listgen_block_dim);

    if (parent->type == SNodeType::dense) {  // TODO: dynamic
      // dense nodes have no allocator, while they have large branching
      // factors. therefore we use a for loop to enumerate the elements

      // we need a for loop here since parent->get_max_n() may be bigger than
      // max_gpu_block_size
      emit("auto cid = div / child_block_division + input_meta.start_loop;");
      emit("if (cid >= input_meta.end_loop) break;");
    } else {
      emit("auto cid = 0;");
    }

    if (parent->type == SNodeType::dense) {
      emit("auto {} = cid;", loopgen.loop_variable(parent));
      loopgen.update_indices(parent);
      loopgen.single_loop_body_head(snode);
    }

    // check if necessary
    emit("int start_idx = div % child_block_division * {};", child_block_size);
    emit("int end_idx = (div % child_block_division + 1) * {};",
         child_block_size);

    if (snode->type == SNodeType::dynamic) {
      emit("if (start_idx >= {}_cache->get_n()) break;", snode->node_type_name);
      emit("end_idx = min(end_idx, {}_cache->get_n());", snode->node_type_name);
    }

    emit(
        "int meta_id = atomicAdd((unsigned long long *)(&{}->resident_tail), "
        "1ULL);",
        leaf_allocator);
    // emit(R"(printf("{} %d\n", meta_id);)", snode->node_type_name);
    emit("auto &meta = {}->resident_pool[meta_id];", leaf_allocator);

    for (int i = 0; i < max_num_indices; i++)
      emit("meta.indices[{}] = {};", i, loopgen.index_name_global(parent, i));

    emit("meta.ptr = {}_cache;", snode->node_type_name);
    emit("meta.start_loop = start_idx;");
    emit("meta.end_loop = end_idx;");
    emit("}}");

    emit("}}");
    emit("}}");

    emit("");

    // host function
    emit("void {}(Context context) {{", listgen_func_name(snode));
    emit("backup_tails<{}><<<1, 1>>>();", parent->node_type_name);
    emit("reset_execution_tail<{}><<<1, 1>>>();", parent->node_type_name);
    emit("reset_tails<{}><<<1, 1>>>();", snode->node_type_name);
    emit("{}_listgen_device<<<{}, {}>>>(context);", snode->node_type_name,
         grid_dim, listgen_block_dim);
    emit("}}");
  }

  std::string listgen_func_name(SNode *leaf) {
    return fmt::format("{}_listgen", leaf->node_type_name);
  }

  void struct_for_new(Stmt *for_stmt_) {
    // struct for
    TC_ASSERT_INFO(current_struct_for == nullptr,
                   "Struct for cannot be nested.");
    auto for_stmt = for_stmt_->as<StructForStmt>();
    current_struct_for = for_stmt;
    auto leaf = for_stmt->snode->parent;

    if (for_stmt->block_size == 0) {
      for_stmt->block_size =
          std::min(1 << leaf->total_num_bits, max_gpu_block_size);
    }

    TC_ASSERT((1 << leaf->total_num_bits) % for_stmt->block_size == 0);
    int block_division = (1 << leaf->total_num_bits) / for_stmt->block_size;

    std::vector<SNode *> path;
    for (auto p = leaf; !p->has_allocator(); p = p->parent) {
      path.push_back(p);
    }

    emit_listgen_func(leaf, block_division);

    for (int i = 1; i < path.size(); i++) {
      emit_listgen_func(path[i]);
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
      emit("__syncthreads();");
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
      emit("if ({} < leaves[leaf_loop].end_loop)", loopgen.loop_variable(leaf),
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

    emit("}}");  // end for
    emit("}}");  // end kernel

    emit("extern \"C\" void {} (Context context) {{\n", codegen->func_name);
    emit("auto root = ({} *)context.buffers[0];",
         current_program->snode_root->node_type_name);
    emit("{{");

    if (debug)
      emit("auto t = get_time();");

    std::string vars;
    for (int i = 0; i < for_stmt->loop_vars.size(); i++) {
      vars += for_stmt->loop_vars[i]->raw_name();
      if (i + 1 < for_stmt->loop_vars.size()) {
        vars += ",";
      }
    }
    emit("gpu_runtime_init();");
    emit("int blockDim = ({}::get_max_n()+ {} - 1) / {};", leaf->node_type_name,
         block_division, block_division);
    emit("");

    // generate the list
    emit(R"(GPUProfiler::get_instance().start("{}_list_gen");)",
         codegen->func_name);

    std::reverse(path.begin(), path.end());
    for (auto &s : path) {
      emit("{}(context);", listgen_func_name(s));
    }
    emit(R"(GPUProfiler::get_instance().stop();)");

    emit("");

    if (debug) {
      emit("cudaDeviceSynchronize();");
      emit(
          R"(printf("task list %d\n", Managers::get_allocator<{}>()->resident_tail);)",
          leaf->node_type_name);
      emit(
          R"(printf("kernel {} <<<%d, %d>>> \n", {}, blockDim);)",
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
    emit(R"(GPUProfiler::get_instance().start("{}");)", codegen->func_name);
    emit("{}_kernel<<<{}, blockDim>>>(context);", codegen->func_name, grid_dim);
    emit(R"(GPUProfiler::get_instance().stop();)");
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

  // For cases where the kernel body has only a for loop
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
      emit(R"(GPUProfiler::get_instance().start("{}");)", codegen->func_name);
      emit("{}_kernel<<<{}, {}>>>(context);", codegen->func_name, num_blocks,
           block_size);
      emit(R"(GPUProfiler::get_instance().stop();)");
    } else {
      auto struct_for = for_stmt_->as<StructForStmt>();
      struct_for_new(for_stmt_.get());
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
    std::string ns;
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
      emit("printf(\"[debug] {}\" \" = %d\\n\", {});", print_stmt->str,
           print_stmt->stmt->raw_name());
    } else if (print_stmt->stmt->ret_type.data_type == DataType::f32) {
      emit("printf(\"[debug] {}\" \" = %f\\n\", {});", print_stmt->str,
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
                   "StructFor cannot be nested.");
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
    auto snode = stmt->snodes[0];
    if (current_scratch_pads && current_scratch_pads->has(snode)) {
      return;
    }

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
    if (stmt->op_type != SNodeOpType::activate &&
        stmt->op_type != SNodeOpType::probe)
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
      TC_NOT_IMPLEMENTED;
    }
    emit("}}");
  }

  void visit(GlobalStoreStmt *stmt) {
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

  void visit(GlobalLoadStmt *stmt) {
    TC_ASSERT(stmt->width() == 1);
    if (stmt->ptr->is<GlobalPtrStmt>()) {
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
    auto init = stmt->elements.serialize(
        [&](const VectorElement &elem) {
          TC_ASSERT(elem.index == 0);
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

  void visit(RangeAssumptionStmt *stmt) {
    /*
    emit("TC_ASSERT({} + {} <= {});", stmt->base->name(), stmt->low,
         stmt->input->name());
    emit("TC_ASSERT({} < {} + {});", stmt->input->name(), stmt->base->name(),
         stmt->high);
    */
    emit("const auto {} = {};", stmt->raw_name(), stmt->input->raw_name());
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
    for (int i = 0; i < stmt->inputs.size(); i++) {
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
  irpass::simplify(ir);
  irpass::re_id(ir);
  if (prog->config.print_ir)
    irpass::print(ir);
  if (prog->config.lower_access) {
    irpass::lower_access(ir);
    if (prog->config.print_ir) {
      TC_TRACE("Access Lowered:");
      irpass::re_id(ir);
      irpass::print(ir);
    }
    for (int i = 0; i < 1; i++) {
      irpass::die(ir);
      if (prog->config.print_ir) {
        TC_TRACE("DIEd:");
        irpass::re_id(ir);
        irpass::print(ir);
      }
      irpass::simplify(ir);
      if (prog->config.print_ir) {
        TC_TRACE("DupEliminated2:");
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

#include "gpu_old.h"
