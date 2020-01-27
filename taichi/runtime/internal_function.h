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
  auto ptr = (ListManager *)allocate_aligned((Runtime *)context->runtime,
                                             sizeof(ListManager), 4096);
  list = new (ptr) ListManager(context, 4, 16);
  for (int i = 0; i < 320; i++) {
    printf("appending %d\n", i);
    list->append(&i);
  }
  return 0;
}
