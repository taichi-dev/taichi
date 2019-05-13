#include "loopgen.h"
#include <cuda_runtime.h>

TLANG_NAMESPACE_BEGIN

LoopGenerator::LoopGenerator(taichi::Tlang::CodeGenBase *gen) : gen(gen) {
  int num_SMs;
  cudaDeviceGetAttribute(&num_SMs, cudaDevAttrMultiProcessorCount, 0);
  grid_dim = num_SMs * 32;  // each SM can have 16-32 resident blocks
}

void LoopGenerator::emit_listgen_func(SNode *snode,
                                      int child_block_division,
                                      std::string suffix) {
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
  emit("__global__ void {}_listgen{}_device(Context context) {{",
       snode->node_type_name, suffix);

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
  emit("auto num_leaves = Managers::get_allocator<{}>()->resident_tail_const;",
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
           index_name_global(parent->parent, i), i);
    }
  } else {
    emit("auto list_element = ({}::child_type *)leaves[leaf_loop].ptr;",
         parent->node_type_name);
    auto chid = parent->child_id(snode);
    TC_ASSERT(chid != -1);
    emit("auto {}_cache = list_element->get{}();", snode->node_type_name, chid);
    for (int i = 0; i < max_num_indices; i++) {
      emit("auto {} = leaves[leaf_loop].indices[{}];",
           index_name_global(parent, i), i);
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
    emit("if (!{}_cache->is_active(cid)) continue;", parent->node_type_name);
  } else {
    emit("auto cid = 0;");
  }

  if (parent->type == SNodeType::dense) {
    emit("auto {} = cid;", loop_variable(parent));
    update_indices(parent);
    single_loop_body_head(snode);
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
    emit("meta.indices[{}] = {};", i, index_name_global(parent, i));

  emit("meta.ptr = {}_cache;", snode->node_type_name);
  emit("meta.start_loop = start_idx;");
  emit("meta.end_loop = end_idx;");
  emit("}}");

  emit("}}");
  emit("}}");

  emit("");

  // host function
  emit("void {}(Context context) {{", listgen_func_name(snode, suffix));
  emit("backup_tails<{}><<<1, 1>>>();", parent->node_type_name);
  emit("reset_execution_tail<{}><<<1, 1>>>();", parent->node_type_name);
  emit("reset_tails<{}><<<1, 1>>>();", snode->node_type_name);
  emit("{}_device<<<{}, {}>>>(context);",
       listgen_func_name(snode, suffix), grid_dim, listgen_block_dim);
  emit("}}");
}

TLANG_NAMESPACE_END
