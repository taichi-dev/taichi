#include <atomic>
#include <cstdio>

int glo;
long long tot;
float fglo;

int main() {
  glo = 0;
#pragma omp parallel for
  for (int i = 0; i < 100000000; i++) {
    glo += 1;
  }
  printf("%d\n", glo);

  glo = 0;
  tot = 0;
#pragma omp parallel for
  for (int i = 0; i < 100000000; i++) {
    auto v =
        __atomic_add_fetch(&glo, 1, std::memory_order::memory_order_relaxed);
    __atomic_add_fetch(&tot, v, std::memory_order_relaxed);
  }
  printf("glo %d\n", glo);
  printf("tot %lld\n", tot);

  glo = 0;
  tot = 0;
#pragma omp parallel for
  for (int i = 0; i < 100000000; i++) {
    auto v =
        __atomic_add_fetch(&glo, 1, std::memory_order::memory_order_seq_cst);
    __atomic_add_fetch(&tot, v, std::memory_order_seq_cst);
  }
  printf("glo %d\n", glo);
  printf("tot %lld\n", tot);

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
