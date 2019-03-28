#include <atomic>
#include <cstdio>

int glo;
float fglo;

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

  /*
  fglo = 0;
#pragma omp parallel for
  for (int i = 0; i < 100000000; i++) {
    __atomic_add_fetch(&fglo, 1.0f, std::memory_order::memory_order_seq_cst);
  }
  printf("%f\n", fglo);
   */
  return 0;
}
