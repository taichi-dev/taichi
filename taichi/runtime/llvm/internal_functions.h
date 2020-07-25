// Note that TI_ASSERT provided by runtime doesn't check fail immediately. As
// a hack, we just check the last error code whenever we call TI_ASSERT.
#define TI_TEST_CHECK(cond, r)                               \
  do {                                                       \
    TI_ASSERT(cond);                                         \
    if ((r)->error_code) {                                   \
      taichi_printf((r), "%s", (r)->error_message_template); \
      abort();                                               \
    }                                                        \
  } while (0)

i32 do_nothing(Context *context) {
  return 0;
}

i32 refresh_counter(Context *context) {
  auto runtime = context->runtime;
  auto queue = runtime->mem_req_queue;
  queue->tail++;
  return 0;
}

i32 test_stack(Context *context) {
  auto stack = new u8[132];
  stack_push(stack, 16, 4);
  stack_push(stack, 16, 4);
  stack_push(stack, 16, 4);
  stack_push(stack, 16, 4);
  return 0;
}

i32 test_list_manager(Context *context) {
  auto runtime = context->runtime;
  taichi_printf(runtime, "LLVMRuntime %p\n", runtime);
  auto list = context->runtime->create<ListManager>(runtime, 4, 16);
  for (int i = 0; i < 320; i++) {
    taichi_printf(runtime, "appending %d\n", i);
    auto j = i + 5;
    list->append(&j);
  }
  for (int i = 0; i < 320; i++) {
    TI_TEST_CHECK(list->get<i32>(i) == i + 5, runtime);
  }
  return 0;
}

i32 test_node_allocator(Context *context) {
  auto runtime = context->runtime;
  taichi_printf(runtime, "LLVMRuntime %p\n", runtime);
  auto nodes = context->runtime->create<NodeManager>(runtime, sizeof(i64), 4);
  Ptr ptrs[24];
  for (int i = 0; i < 19; i++) {
    taichi_printf(runtime, "allocating %d\n", i);
    ptrs[i] = nodes->allocate();
    taichi_printf(runtime, "ptr %p\n", ptrs[i]);
  }
  for (int i = 0; i < 5; i++) {
    taichi_printf(runtime, "deallocating %d\n", i);
    taichi_printf(runtime, "ptr %p\n", ptrs[i]);
    nodes->recycle(ptrs[i]);
  }
  nodes->gc_serial();
  for (int i = 19; i < 24; i++) {
    taichi_printf(runtime, "allocating %d\n", i);
    ptrs[i] = nodes->allocate();
  }
  for (int i = 5; i < 19; i++) {
    TI_TEST_CHECK(nodes->locate(ptrs[i]) == i, runtime);
  }

  for (int i = 19; i < 24; i++) {
    auto idx = nodes->locate(ptrs[i]);
    taichi_printf(runtime, "i %d", i);
    taichi_printf(runtime, "idx %d", idx);
    TI_TEST_CHECK(idx == i - 19, runtime);
  }
  return 0;
}

i32 test_node_allocator_gc_cpu(Context *context) {
  auto runtime = context->runtime;
  taichi_printf(runtime, "LLVMRuntime %p\n", runtime);
  auto nodes = context->runtime->create<NodeManager>(runtime, sizeof(i64), 4);
  constexpr int kN = 24;
  constexpr int kHalfN = kN / 2;
  Ptr ptrs[kN];
  // Initially |free_list| is empty
  TI_TEST_CHECK(nodes->free_list->size() == 0, runtime);
  for (int i = 0; i < kN; i++) {
    taichi_printf(runtime, "[1] allocating %d\n", i);
    ptrs[i] = nodes->allocate();
    taichi_printf(runtime, "[1] ptr %p\n", ptrs[i]);
  }
  for (int i = 0; i < kN; i++) {
    taichi_printf(runtime, "[1] deallocating %d\n", i);
    taichi_printf(runtime, "[1] ptr %p\n", ptrs[i]);
    nodes->recycle(ptrs[i]);
  }
  TI_TEST_CHECK(nodes->free_list->size() == 0, runtime);
  nodes->gc_serial();
  // After the first round GC, |free_list| should have |kN| items.
  TI_TEST_CHECK(nodes->free_list->size() == kN, runtime);

  // In the second round, all items should come from |free_list|.
  for (int i = 0; i < kHalfN; i++) {
    taichi_printf(runtime, "[2] allocating %d\n", i);
    ptrs[i] = nodes->allocate();
    taichi_printf(runtime, "[2] ptr %p\n", ptrs[i]);
  }
  TI_TEST_CHECK(nodes->allocated_elements == kHalfN, runtime);
  for (int i = 0; i < kHalfN; i++) {
    taichi_printf(runtime, "[2] deallocating %d\n", i);
    taichi_printf(runtime, "[2] ptr %p\n", ptrs[i]);
    nodes->recycle(ptrs[i]);
  }
  nodes->gc_serial();
  // After GC, all items should be returned to |free_list|.
  taichi_printf(runtime, "free_list_size=%d\n", nodes->free_list->size());
  TI_TEST_CHECK(nodes->free_list->size() == kN, runtime);

  return 0;
}

i32 test_active_mask(Context *context) {
  auto rt = context->runtime;
  taichi_printf(rt, "%d activemask %x\n", thread_idx(), cuda_active_mask());

  auto active_mask = cuda_active_mask();
  auto remaining = active_mask;
  while (remaining) {
    auto leader = cttz_i32(remaining);
    taichi_printf(rt, "currnet leader %d bid %d tid %d\n", leader, block_idx(),
                  thread_idx());
    warp_barrier(active_mask);
    remaining &= ~(1u << leader);
  }

  return 0;
}

i32 test_shfl(Context *context) {
  auto rt = context->runtime;
  auto s =
      cuda_shfl_down_sync_i32(cuda_active_mask(), warp_idx() + 1000, 2, 31);
  taichi_printf(rt, "tid %d tid_shfl %d\n", thread_idx(), s);

  return 0;
}

#undef TI_TEST_CHECK
