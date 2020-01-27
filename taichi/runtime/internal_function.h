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
  ListManager list(context, 4, 16);
  for (int i = 0; i < 32; i++) {
    list.append(&i);
  }
  return 0;
}
