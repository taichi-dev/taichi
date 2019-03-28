#include <atomic>
#include <cstdio>

int glo;

int main() {
  glo = 0;
#pragma omp parallel for
  for (int i = 0; i < 100000000; i++) {
    glo += 1;
  }
  printf("%d\n", glo);

  glo = 0;
#pragma omp parallel for
  for (int i = 0; i < 100000000; i++) {
    __atomic_add_fetch(&glo, 1, std::memory_order::memory_order_seq_cst);
  }
  printf("%d\n", glo);
  return 0;
}
