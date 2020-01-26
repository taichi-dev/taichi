i32 do_nothing(Context *context) {
  return 0;
}

i32 refresh_counter(Context *context) {
  auto runtime = (Runtime *)context->runtime;
  auto queue = runtime->mem_req_queue;
  queue->tail++;
  return 0;
}
