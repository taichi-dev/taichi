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
    TI_ASSERT(list->get<i32>(i) == i + 5);
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
    TI_ASSERT(nodes->locate(ptrs[i]) == i);
  }

  for (int i = 19; i < 24; i++) {
    auto idx = nodes->locate(ptrs[i]);
    taichi_printf(runtime, "i %d", i);
    taichi_printf(runtime, "idx %d", idx);
    TI_ASSERT(idx == i - 19);
  }
  return 0;
}
