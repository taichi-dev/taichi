extern "C" {

int thread_idx() {
  return 0;
}

int warp_size() {
  return 32;
}

int warp_idx() {
  return thread_idx() % warp_size();
}

int block_idx() {
  return 0;
}

int block_dim() {
  return 0;
}

int grid_dim() {
  return 0;
}
}
