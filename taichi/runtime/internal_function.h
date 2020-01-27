i32 do_nothing(Context *context) {
  return 0;
}

i32 refresh_counter(Context *context) {
  auto runtime = (Runtime *)context->runtime;
  auto queue = runtime->mem_req_queue;
  queue->tail++;
  return 0;
}

i32 test_list_manager(Context *context) {
  ListManager *list;
  auto runtime = (Runtime *)context->runtime;
  Printf("Runtime %p\n", runtime);
  auto ptr = (ListManager *)(Runtime *)runtime->request_allocate_aligned(
      sizeof(ListManager), 4096);
  list = new (ptr) ListManager((Runtime *)context->runtime, 4, 16);
  for (int i = 0; i < 320; i++) {
    Printf("appending %d\n", i);
    auto j = i + 5;
    list->append(&j);
  }
  for (int i = 0; i < 320; i++) {
    TC_ASSERT(*(i32 *)list->get(i) == i + 5);
  }
  return 0;
}
