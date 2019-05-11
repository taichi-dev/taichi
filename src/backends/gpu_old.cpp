#include <cuda_runtime.h>
#include "gpu.h"
#include "loop_gen.h"
#include "../scratch_pad.h"

TLANG_NAMESPACE_BEGIN

void GPUCodeGen::struct_for_old(Stmt *for_stmt_) {
  TC_WARN("Using old struct for");
  // struct for
  TC_ASSERT_INFO(current_struct_for == nullptr, "Struct for cannot be nested.");
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
  // always sync here since CPU list gen needs the latest data structure
  emit("cudaDeviceSynchronize();");
  emit("auto root = ({} *)context.buffers[0];",
       current_program->snode_root->node_type_name);
  emit("{{");

  if (debug)
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
  if (debug) {
    emit("auto list_time = (get_time() - t) * 1000;");
    emit(R"(printf("kernel {} <<<%d, %d>>> ", gridDim, blockDim);)",
         codegen->func_name);
    emit(R"(std::cout << "list gen: " << list_time << " ms  ";)");
    emit("cudaEvent_t start, stop;");

    if (current_program->get_current_kernel().benchmarking) {
      emit("while(1) {{");
    }

    emit("cudaEventCreate(&start);");
    emit("cudaEventCreate(&stop);");
    emit("cudaEventRecord(start);");
  }
  emit(R"(GPUProfiler::get_instance().start("{}");)", codegen->func_name);
  emit("{}_kernel<<<gridDim, blockDim>>>(context);", codegen->func_name);
  emit(R"(GPUProfiler::get_instance().stop();)");
  emit("cudaDeviceSynchronize();");
  if (debug) {
    emit("cudaEventRecord(stop);");
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
  }

  emit("cudaFree(context.leaves); context.leaves = nullptr;");
  emit("}}");
  current_struct_for = nullptr;
}

TLANG_NAMESPACE_END
